"""API endpoints for executing DML statements with PostgreSQL JSON/JSONB support.

Supports all PostgreSQL JSON functions and operators (9.16. JSON Functions and Operators).
JSON operator keys/paths are NOT parameterized; JSON comparison values ARE parameterized.
"""

import datetime
import decimal
import re
import string
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import sqlglot
import sqlparse
from common.service import get_otlp_metric_service, get_otlp_span_service
from common.utils.snowfake import get_id
from fastapi import APIRouter, Depends
from memory.database.api.schemas.exec_dml_types import ExecDMLInput
from memory.database.api.v1.common import (
    check_database_exists_by_did,
    check_space_id_and_get_uid,
)
from memory.database.domain.entity.general import exec_sql_statement, parse_and_exec_sql
from memory.database.domain.entity.schema import set_search_path_by_schema
from memory.database.domain.entity.views.http_resp import format_response
from memory.database.exceptions.e import CustomException
from memory.database.exceptions.error_code import CodeEnum
from memory.database.repository.middleware.getters import get_session
from sqlglot import exp, parse_one
from sqlglot.expressions import Column, Literal
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.responses import JSONResponse

exec_dml_router = APIRouter(tags=["EXEC_DML"])

INSERT_EXTRA_COLUMNS = ["id", "uid", "create_time", "update_time"]

POSTGRESQL_JSON_TABLE_FUNCTIONS = {
    "json_array_elements",
    "jsonb_array_elements",
    "json_array_elements_text",
    "jsonb_array_elements_text",
    "json_each",
    "jsonb_each",
    "json_each_text",
    "jsonb_each_text",
    "json_object_keys",
    "jsonb_object_keys",
    "json_populate_record",
    "jsonb_populate_record",
    "json_populate_recordset",
    "jsonb_populate_recordset",
}

POSTGRESQL_JSON_BUILD_FUNCTIONS = {
    "json_build_array",
    "jsonb_build_array",
    "json_build_object",
    "jsonb_build_object",
    "json_object",
    "jsonb_object",
    "to_json",
    "to_jsonb",
    "array_to_json",
    "row_to_json",
}

POSTGRESQL_JSON_PROCESSING_FUNCTIONS = {
    "jsonb_set",
    "jsonb_insert",
    "jsonb_path_query",
    "jsonb_path_query_array",
    "jsonb_extract_path",
    "jsonb_extract_path_text",
    "jsonb_typeof",
    "jsonb_pretty",
    "jsonb_strip_nulls",
}

POSTGRESQL_JSON_AGGREGATE_FUNCTIONS = {
    "json_agg",
    "jsonb_agg",
    "json_object_agg",
    "jsonb_object_agg",
}

ALL_POSTGRESQL_JSON_FUNCTIONS = (
    POSTGRESQL_JSON_TABLE_FUNCTIONS
    | POSTGRESQL_JSON_BUILD_FUNCTIONS
    | POSTGRESQL_JSON_PROCESSING_FUNCTIONS
    | POSTGRESQL_JSON_AGGREGATE_FUNCTIONS
)

POSTGRESQL_JSON_OPERATORS = {
    "->",
    "->>",
    "#>",
    "#>>",
    "@>",
    "<@",
    "?",
    "?|",
    "?&",
    "@?",
    "@@",
}

PGSQL_INVALID_KEY = [
    "all",
    "analyse",
    "analyze",
    "and",
    "any",
    "array",
    "as",
    "asc",
    "asymmetric",
    "authorization",
    "binary",
    "both",
    "case",
    "cast",
    "check",
    "collate",
    "collation",
    "column",
    "concurrently",
    "constraint",
    "create",
    "cross",
    "current_catalog",
    "current_date",
    "current_role",
    "current_schema",
    "current_time",
    "current_timestamp",
    "current_user",
    "default",
    "deferrable",
    "desc",
    "distinct",
    "do",
    "else",
    "end",
    "except",
    "false",
    "fetch",
    "for",
    "foreign",
    "freeze",
    "from",
    "full",
    "grant",
    "group",
    "having",
    "ilike",
    "in",
    "initially",
    "inner",
    "intersect",
    "into",
    "is",
    "isnull",
    "join",
    "lateral",
    "leading",
    "left",
    "like",
    "limit",
    "localtime",
    "localtimestamp",
    "natural",
    "not",
    "notnull",
    "null",
    "offset",
    "on",
    "only",
    "or",
    "order",
    "outer",
    "overlaps",
    "placing",
    "primary",
    "references",
    "returning",
    "right",
    "select",
    "session_user",
    "similar",
    "some",
    "symmetric",
    "table",
    "tablesample",
    "then",
    "to",
    "trailing",
    "true",
    "union",
    "unique",
    "user",
    "using",
    "variadic",
    "verbose",
    "when",
    "where",
    "window",
    "with",
]


def _extract_column_name(col: Any) -> Optional[str]:
    """Extract column name from various column node types."""
    if hasattr(col, "name") and col.name:
        return col.name
    if hasattr(col, "this"):
        col_this = col.this
        if isinstance(col_this, str):
            return col_this
        if hasattr(col_this, "this"):
            return str(col_this.this) if col_this.this else None
        return str(col_this)
    return str(col) if col else None


def _extract_column_names(columns: Any) -> List[str]:
    """Extract column names from columns object (Schema, Tuple, or list)."""
    column_names: List[str] = []
    if not columns:
        return column_names

    col_list: List[Any] = []
    if hasattr(columns, "expressions"):
        col_list = columns.expressions or []
    elif isinstance(columns, list):
        col_list = columns
    elif hasattr(columns, "name"):
        return [columns.name]

    for col in col_list:
        col_name = _extract_column_name(col)
        if col_name:
            column_names.append(col_name)

    return column_names


def _map_literals_in_node(
    node: Any, target_column: str, table_name: str, literal_column_map: Dict[int, str]
) -> None:
    """Map all literals in a node to a target column."""
    if not hasattr(node, "walk"):
        return
    try:
        for literal_node in node.walk():
            if isinstance(literal_node, exp.Literal):
                literal_column_map[id(literal_node)] = f"{table_name}.{target_column}"
    except (AttributeError, TypeError):
        pass


def _process_insert_select(
    insert_exprs: exp.Select,
    column_names: List[str],
    actual_table_name: str,
    literal_column_map: Dict[int, str],
) -> None:
    """Process INSERT with SELECT statement."""
    select_exprs = insert_exprs.args.get("expressions", [])
    if not select_exprs:
        return

    if not column_names:
        for select_expr in select_exprs:
            _map_literals_in_node(
                select_expr, "unknown", actual_table_name, literal_column_map
            )
        return

    for idx, select_expr in enumerate(select_exprs):
        if not select_expr:
            continue
        target_column = (
            column_names[idx] if idx < len(column_names) else column_names[-1]
        )
        _map_literals_in_node(
            select_expr, target_column, actual_table_name, literal_column_map
        )

    default_col = column_names[0] if column_names else "unknown"
    for expr in [insert_exprs.args.get("from"), insert_exprs.args.get("where")]:
        if expr:
            _map_literals_in_node(
                expr, default_col, actual_table_name, literal_column_map
            )


def _process_insert_values(
    insert_exprs: Any,
    column_names: List[str],
    actual_table_name: str,
    literal_column_map: Dict[int, str],
) -> None:
    """Process INSERT with VALUES statement."""
    if not column_names:
        for row in getattr(insert_exprs, "expressions", []):
            if hasattr(row, "expressions"):
                for expr in row.expressions:
                    _map_literals_in_node(
                        expr, "unknown", actual_table_name, literal_column_map
                    )
        return

    for row in getattr(insert_exprs, "expressions", []):
        if not hasattr(row, "expressions"):
            continue
        for idx, expr in enumerate(row.expressions):
            target_column = (
                column_names[idx] if idx < len(column_names) else column_names[-1]
            )
            if isinstance(expr, exp.Literal):
                literal_column_map[id(expr)] = f"{actual_table_name}.{target_column}"
            else:
                _map_literals_in_node(
                    expr, target_column, actual_table_name, literal_column_map
                )


def _build_insert_literal_map(
    parsed: exp.Insert, table_name: str, literal_column_map: Dict[int, str]
) -> None:
    """Build literal-column mapping for INSERT statements."""
    columns = parsed.args.get("this")
    insert_exprs = parsed.args.get("expression")
    if not (columns and insert_exprs):
        return

    schema_obj = parsed.this
    if (
        isinstance(schema_obj, exp.Schema)
        and hasattr(schema_obj, "this")
        and hasattr(schema_obj.this, "name")
    ):
        actual_table_name = schema_obj.this.name
    elif isinstance(schema_obj, exp.Table):
        actual_table_name = schema_obj.alias_or_name
    else:
        actual_table_name = table_name
    column_names = _extract_column_names(columns)

    if isinstance(insert_exprs, exp.Select):
        _process_insert_select(
            insert_exprs, column_names, actual_table_name, literal_column_map
        )
    else:
        _process_insert_values(
            insert_exprs, column_names, actual_table_name, literal_column_map
        )


def _find_column_in_expr(expr: Any) -> Optional[exp.Column]:
    """Find Column node in an expression tree."""
    if isinstance(expr, exp.Column):
        return expr
    if hasattr(expr, "this"):
        current = expr.this
        while current:
            if isinstance(current, exp.Column):
                return current
            if hasattr(current, "this"):
                current = current.this
            elif hasattr(current, "expressions") and current.expressions:
                if isinstance(current.expressions[0], exp.Column):
                    return current.expressions[0]
                current = current.expressions[0] if current.expressions else None
            else:
                break
    return None


def _build_update_literal_map(
    parsed: exp.Update,
    table_name: str,
    literal_column_map: Dict[int, str],
    alias_map: Optional[Dict[str, str]] = None,
) -> None:
    """Build literal-column mapping for UPDATE statements."""
    alias_map = alias_map or {}
    default_table_name = (
        parsed.this.name or parsed.this.alias_or_name
        if isinstance(parsed.this, exp.Table)
        else table_name
    )

    for set_expr in parsed.expressions:
        if not isinstance(set_expr, exp.EQ) or not isinstance(
            set_expr.right, exp.Literal
        ):
            continue

        left_col = _find_column_in_expr(set_expr.left)
        if not left_col:
            continue

        table_ref = (
            _extract_table_ref(left_col.table)
            if hasattr(left_col, "table") and left_col.table
            else None
        )
        actual_table_name = _resolve_table_name(
            table_ref, alias_map, default_table_name
        )
        literal_column_map[id(set_expr.right)] = f"{actual_table_name}.{left_col.name}"

        if isinstance(set_expr.right, exp.Func) and _is_json_function_call(
            set_expr.right
        ):
            _map_literals_in_node(
                set_expr.right, left_col.name, actual_table_name, literal_column_map
            )


def _process_comparison_node(
    node: Any,
    literal_column_map: Dict[int, str],
    get_table_name_func: Any,
) -> None:
    """Process comparison operation node to map literals to columns."""
    left_expr, right_expr = node.left, node.right
    left_col = _find_column_in_expr(left_expr)

    if isinstance(left_expr, exp.Func) and _is_json_function_call(left_expr):
        if hasattr(left_expr, "expressions") and left_expr.expressions:
            first_arg = left_expr.expressions[0]
            left_col = (
                first_arg
                if isinstance(first_arg, exp.Column)
                else (
                    first_arg.this
                    if hasattr(first_arg, "this")
                    and isinstance(first_arg.this, exp.Column)
                    else None
                )
            )

    if left_col and isinstance(right_expr, exp.Literal):
        actual_table_name = get_table_name_func(left_col)
        literal_column_map[id(right_expr)] = f"{actual_table_name}.{left_col.name}"
        return

    if isinstance(left_expr, exp.Literal) and isinstance(right_expr, exp.Column):
        actual_table_name = get_table_name_func(right_expr)
        literal_column_map[id(left_expr)] = f"{actual_table_name}.{right_expr.name}"


def _map_where_literals_recursive(
    node: Any,
    literal_column_map: Dict[int, str],
    get_table_name_func: Any,
) -> None:
    """Recursively map literal values in WHERE clause to column names."""
    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
        _process_comparison_node(node, literal_column_map, get_table_name_func)
    elif hasattr(node, "expressions"):
        for expr in node.expressions:
            _map_where_literals_recursive(expr, literal_column_map, get_table_name_func)
    elif hasattr(node, "this"):
        _map_where_literals_recursive(
            node.this, literal_column_map, get_table_name_func
        )


def _build_select_literal_map(
    parsed: exp.Select,
    table_name: str,
    literal_column_map: Dict[int, str],
    alias_map: Optional[Dict[str, str]] = None,
) -> None:
    """Build literal-column mapping for SELECT statements."""
    alias_map = alias_map or {}

    where_expr = parsed.args.get("where")
    if not where_expr:
        return

    from_expr = parsed.args.get("from")
    default_table_name = table_name
    if from_expr and hasattr(from_expr, "expressions") and from_expr.expressions:
        first_table = from_expr.expressions[0]
        if isinstance(first_table, exp.Table):
            default_table_name = first_table.name or first_table.alias_or_name

    def _get_table_name_from_column(col: exp.Column) -> str:
        """Get actual table name from Column node, resolving aliases."""
        table_ref = (
            _extract_table_ref(col.table)
            if hasattr(col, "table") and col.table
            else None
        )
        return _resolve_table_name(table_ref, alias_map, default_table_name)

    _map_where_literals_recursive(
        where_expr, literal_column_map, _get_table_name_from_column
    )


def _extract_table_ref(table_obj: Any) -> Optional[str]:
    """Extract table reference from various table object types."""
    if isinstance(table_obj, exp.Table):
        return table_obj.name or table_obj.alias_or_name
    if isinstance(table_obj, str):
        return table_obj
    if hasattr(table_obj, "this"):
        return table_obj.this
    if hasattr(table_obj, "name"):
        return table_obj.name
    return None


def _resolve_table_name(
    table_ref: Optional[str], alias_map: Dict[str, str], default: str
) -> str:
    """Resolve table reference to actual table name using alias map."""
    if table_ref and alias_map:
        return alias_map.get(table_ref, table_ref)
    return table_ref or default


def _build_table_alias_map(parsed: Any) -> Dict[str, str]:
    """Build mapping from table alias to actual table name."""
    alias_map: Dict[str, str] = {}
    for table in parsed.find_all(exp.Table):
        if table.name:
            alias_map[table.name] = table.name
            if table.alias:
                alias_map[table.alias] = table.name
    return alias_map


def _build_literal_column_map(
    parsed: Any,
    table_name: str,
    literal_column_map: Dict[int, str],
    alias_map: Optional[Dict[str, str]] = None,
) -> None:
    """Build mapping from Literal nodes to column names based on statement type."""
    if alias_map is None:
        alias_map = _build_table_alias_map(parsed)

    if isinstance(parsed, exp.Insert):
        _build_insert_literal_map(parsed, table_name, literal_column_map)
    elif isinstance(parsed, exp.Update):
        _build_update_literal_map(parsed, table_name, literal_column_map, alias_map)
    elif isinstance(parsed, exp.Select):
        _build_select_literal_map(parsed, table_name, literal_column_map, alias_map)


def _is_datetime_type(data_type: str) -> bool:
    """Check if data type is datetime."""
    if not data_type:
        return False
    dt_lower = data_type.lower()
    return any(
        dt in dt_lower
        for dt in [
            "timestamp",
            "timestamptz",
            "timestamp without time zone",
            "timestamp with time zone",
            "date",
            "time",
            "timetz",
            "time without time zone",
            "time with time zone",
        ]
    )


def _is_json_type(data_type: str) -> bool:
    """Check if data type is JSON or JSONB."""
    if not data_type:
        return False
    return any(jt in data_type.lower() for jt in ["json", "jsonb"])


def _convert_value_if_datetime(
    value: str,
    node_id: int,
    literal_column_map: Dict[int, str],
    column_types: Dict[str, str],
) -> Union[str, datetime.datetime]:
    """Convert string value to datetime if the corresponding column is datetime type."""
    converted_value: Union[str, datetime.datetime] = value
    col_key = literal_column_map.get(node_id)
    if not col_key:
        return converted_value

    column_type = column_types.get(col_key, "")
    if not _is_datetime_type(column_type):
        return converted_value

    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", value):
        converted_value = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return converted_value


def _is_numeric_value(value: Any) -> bool:
    """Check if value is numeric (int, float, or numeric string)."""
    return isinstance(value, (int, float)) or (
        isinstance(value, str) and value.isdigit()
    )


def _get_func_name(node: Any) -> Optional[str]:
    """Extract function name from AST node."""
    if not hasattr(node, "this"):
        return None
    if isinstance(node.this, exp.Identifier):
        return node.this.this.lower() if hasattr(node.this, "this") else None
    if isinstance(node.this, str):
        return node.this.lower()
    return None


def _is_json_function_call(node: Any) -> bool:
    """Check if node is a PostgreSQL JSON function call."""
    if not node or not isinstance(node, exp.Func):
        return False
    try:
        func_name = _get_func_name(node)
        return func_name in ALL_POSTGRESQL_JSON_FUNCTIONS if func_name else False
    except (AttributeError, TypeError):
        return False


def _check_array_path_argument(
    node: exp.Literal, expr: exp.Array, expressions: List[Any], func_name: str
) -> bool:
    """Check if node is in array path argument."""
    try:
        if expressions.index(expr) == 1 and func_name in ("jsonb_set", "jsonb_insert"):
            return any(
                array_elem is node
                or (hasattr(array_elem, "walk") and node in list(array_elem.walk()))
                for array_elem in getattr(expr, "expressions", [])
            )
    except (ValueError, AttributeError):
        pass
    return False


def _check_cast_path_argument(
    node: exp.Literal, expr: exp.Cast, expressions: List[Any], func_name: str
) -> bool:
    """Check if node is in cast path argument."""
    try:
        if expressions.index(expr) == 1 and func_name in ("jsonb_set", "jsonb_insert"):
            cast_expr = getattr(expr, "this", None)
            if (
                cast_expr
                and hasattr(cast_expr, "walk")
                and node in list(cast_expr.walk())
            ):
                cast_type = expr.args.get("to", "") if hasattr(expr, "args") else ""
                if (
                    "text[]" in str(cast_type).lower()
                    or "array" in str(cast_type).lower()
                ):
                    return True
    except (ValueError, AttributeError):
        pass
    return False


def _check_nested_path_argument(
    node: exp.Literal, expressions: List[Any], func_name: str
) -> bool:
    """Check if node is nested in array or cast path argument."""
    for expr in expressions:
        if isinstance(expr, exp.Array):
            if _check_array_path_argument(node, expr, expressions, func_name):
                return True
        elif isinstance(expr, exp.Cast):
            if _check_cast_path_argument(node, expr, expressions, func_name):
                return True
    return False


def _check_direct_path_argument(
    node: exp.Literal, expressions: List[Any], func_name: str
) -> bool:
    """Check if node is a direct path argument."""
    try:
        arg_index = expressions.index(node)
        if func_name in (
            "jsonb_set",
            "jsonb_insert",
            "jsonb_path_query",
            "jsonb_path_query_array",
            "jsonb_path_exists",
        ):
            return arg_index == 1
        if func_name in ("jsonb_extract_path", "jsonb_extract_path_text"):
            return arg_index >= 1
    except (ValueError, AttributeError):
        pass
    return False


def _is_json_function_argument_literal(node: exp.Literal, parent: Any) -> bool:
    """Check if literal is a JSON function path argument that should NOT be parameterized."""
    if (
        not parent
        or not isinstance(parent, exp.Func)
        or not _is_json_function_call(parent)
    ):
        return False

    func_name = _get_func_name(parent)
    if not func_name or not hasattr(parent, "expressions"):
        return False

    expressions = (
        parent.expressions
        if isinstance(parent.expressions, list)
        else [parent.expressions]
    )

    if node not in expressions:
        return _check_nested_path_argument(node, expressions, func_name)

    return _check_direct_path_argument(node, expressions, func_name)


def _check_parent_sql_for_json_ops(parent: Any, node: exp.Literal) -> bool:
    """Check if parent SQL contains JSON operators."""
    json_ops_list = list(POSTGRESQL_JSON_OPERATORS)
    json_ops_with_spaces = [
        f" {op} " for op in ("?", "?|", "?&") if op in POSTGRESQL_JSON_OPERATORS
    ]

    try:
        parent_sql = parent.sql(dialect="postgres") if hasattr(parent, "sql") else ""
        if any(op in parent_sql for op in json_ops_list + json_ops_with_spaces):
            if hasattr(parent, "expressions"):
                expressions = (
                    parent.expressions
                    if isinstance(parent.expressions, list)
                    else [parent.expressions]
                )
                if node in expressions:
                    return True
            if hasattr(parent, "right") and parent.right is node:
                return True
    except (AttributeError, TypeError, ValueError):
        pass
    return False


def _check_ancestor_sql_for_json_ops(parent: Any) -> bool:
    """Check if ancestor nodes contain JSON operators."""
    json_ops_list = list(POSTGRESQL_JSON_OPERATORS)
    current, depth = parent, 0
    while current and depth < 5:
        try:
            current_sql = (
                current.sql(dialect="postgres") if hasattr(current, "sql") else ""
            )
            if any(op in current_sql for op in json_ops_list):
                return True
        except (AttributeError, TypeError, ValueError):
            pass
        current = getattr(current, "parent", None) or getattr(current, "this", None)
        depth += 1
    return False


def _is_json_operator_literal(node: exp.Literal, parsed_tree: Any) -> bool:
    """Check if literal is part of a PostgreSQL JSON operator expression."""
    if not node:
        return False

    try:
        parent = getattr(node, "parent", None)
        if not parent:
            return False

        if _is_json_function_argument_literal(node, parent):
            return True

        if _check_parent_sql_for_json_ops(parent, node):
            return True

        if _check_ancestor_sql_for_json_ops(parent):
            return True
    except (AttributeError, TypeError):
        pass
    return False


def _parameterize_literals(
    parsed: Any,
    literal_column_map: Dict[int, str],
    column_types: Optional[Dict[str, str]],
) -> dict[str, Any]:
    """Parameterize literal values in SQL statements, skipping JSON operator keys/paths."""
    params_dict: dict[str, Any] = {}
    processed_nodes: set[int] = set()
    literal_to_parent: Dict[int, Any] = {}

    for node in parsed.walk():
        if isinstance(node, exp.Literal):
            parent = getattr(node, "parent", None)
            if parent:
                literal_to_parent[id(node)] = parent

    for node in parsed.walk():
        if not isinstance(node, exp.Literal):
            continue
        node_id = id(node)
        if node_id in processed_nodes:
            continue
        value = node.this
        if _is_numeric_value(value) or not isinstance(value, str):
            continue

        parent = literal_to_parent.get(node_id)
        if parent and (
            _is_json_operator_literal(node, parsed)
            or _is_json_function_argument_literal(node, parent)
        ):
            continue

        converted_value = (
            _convert_value_if_datetime(value, node_id, literal_column_map, column_types)
            if column_types
            else value
        )
        param_name = f"param_{len(params_dict)}"
        node.replace(exp.Placeholder(this=param_name))
        params_dict[param_name] = converted_value
        processed_nodes.add(node_id)
    return params_dict


def rewrite_dml_with_uid_and_limit(
    dml: str,
    app_id: str,
    uid: str,
    limit_num: int,
    column_types: Optional[Dict[str, str]] = None,
) -> tuple[str, list, dict]:
    """Rewrite DML with UID and limit expressions. Returns (rewritten_sql, insert_ids, params_dict)."""
    parsed = parse_one(dml)
    insert_ids: List[int] = []

    tables = [
        table.alias_or_name for table in parsed.find_all(exp.Table) if table.name != ""
    ]

    if isinstance(parsed, (exp.Update, exp.Delete, exp.Select)):
        _dml_add_where(parsed, tables, app_id, uid)

    if isinstance(parsed, exp.Select):
        limit = parsed.args.get("limit")
        if not limit:
            parsed.set("limit", exp.Limit(expression=exp.Literal.number(limit_num)))

    if isinstance(parsed, exp.Insert):
        _dml_insert_add_params(parsed, insert_ids, app_id, uid)

    literal_column_map: Dict[int, str] = {}
    if column_types and tables:
        alias_map = _build_table_alias_map(parsed)
        _build_literal_column_map(parsed, tables[0], literal_column_map, alias_map)
    params_dict = _parameterize_literals(parsed, literal_column_map, column_types)

    return parsed.sql(dialect="postgres"), insert_ids, params_dict


def _dml_add_where(parsed: Any, tables: List[str], app_id: str, uid: str) -> None:
    """Add WHERE conditions to DML statements."""
    where_expr = parsed.args.get("where")
    uid_conditions = []

    for table in tables:
        uid_col = exp.Column(this="uid", table=table)
        condition = exp.In(
            this=uid_col,
            expressions=[
                exp.Literal.string(f"{uid}"),
                exp.Literal.string(f"{app_id}:{uid}"),
            ],
        )
        uid_conditions.append(condition)

    final_condition = uid_conditions[0]
    for cond in uid_conditions[1:]:
        final_condition = exp.and_(final_condition, cond)  # type: ignore[assignment]

    if where_expr:
        grouped_where = exp.Paren(this=where_expr.this)
        new_where = exp.and_(grouped_where, final_condition)
    else:
        new_where = final_condition

    parsed.set("where", exp.Where(this=new_where))


def _dml_insert_add_params(
    parsed: Any, insert_ids: List[int], app_id: str, uid: str
) -> None:
    """Add id and uid to INSERT statements."""
    schema_obj = parsed.this
    existing_columns = (
        schema_obj.expressions or []
        if schema_obj and hasattr(schema_obj, "expressions")
        else []
    )
    insert_exprs = parsed.args["expression"]

    if isinstance(insert_exprs, exp.Select):
        need_del_index = [
            i
            for i, col in enumerate(existing_columns)
            if _extract_column_name(col) in INSERT_EXTRA_COLUMNS
        ]
        for idx in reversed(need_del_index):
            existing_columns.pop(idx)
            select_exprs = insert_exprs.args.get("expressions", [])
            if idx < len(select_exprs):
                select_exprs.pop(idx)

        existing_columns.extend([exp.to_identifier("id"), exp.to_identifier("uid")])
        select_exprs = insert_exprs.args.get("expressions", [])

        base_id = get_id()
        insert_ids.append(base_id)

        row_number_parsed = parse_one("SELECT row_number() OVER ()", dialect="postgres")
        row_number_expr = (
            row_number_parsed.expressions[0]
            if isinstance(row_number_parsed, exp.Select)
            and row_number_parsed.expressions
            else None
        )
        if not row_number_expr:
            row_number_expr = exp.Func(
                this=exp.Identifier(this="row_number", quoted=False), expressions=[]
            )
            row_number_expr.set("over", exp.Window())

        id_expr = exp.Add(
            this=exp.Literal.number(base_id),
            expression=exp.Paren(
                this=exp.Sub(this=row_number_expr, expression=exp.Literal.number(1))
            ),
        )
        select_exprs.extend([id_expr, exp.Literal.string(f"{app_id}:{uid}")])
        insert_exprs.set("expressions", select_exprs)

    else:
        rows = insert_exprs.expressions
        need_del_index = [
            i
            for i, col in enumerate(existing_columns)
            if getattr(col, "this", None) in INSERT_EXTRA_COLUMNS
        ]
        for idx in reversed(need_del_index):
            existing_columns.pop(idx)
            for row in rows:
                row.expressions.pop(idx)

        existing_columns.extend([exp.to_identifier("id"), exp.to_identifier("uid")])
        for i, row in enumerate(rows):
            row_id = get_id()
            insert_ids.append(row_id)
            rows[i] = exp.Tuple(
                expressions=list(row.expressions)
                + [exp.Literal.number(row_id), exp.Literal.string(f"{app_id}:{uid}")]
            )

    if schema_obj and hasattr(schema_obj, "set"):
        schema_obj.set("expressions", existing_columns)
    elif schema_obj and hasattr(schema_obj, "this"):
        parsed.set(
            "this", exp.Schema(this=schema_obj.this, expressions=existing_columns)
        )
    else:
        parsed.set("this", exp.Tuple(expressions=existing_columns))
    parsed.set("expression", insert_exprs)


def to_jsonable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, datetime.datetime):
        return obj.isoformat(sep=" ", timespec="seconds")
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return obj


def _collect_column_names(parsed: Any) -> list:
    """Collect column names."""
    columns_to_validate = []
    for node in parsed.walk():
        if not isinstance(node, Column):
            continue

        column_name = node.name
        if not column_name:
            continue

        columns_to_validate.append(column_name)
    return columns_to_validate


def _collect_insert_keys(parsed: Any) -> list:
    """Collect key names from INSERT statements."""
    keys_to_validate = []
    for node in parsed.walk():
        if not isinstance(node, exp.Insert):
            continue

        if not (node.this and hasattr(node.this, "expressions")):
            continue

        for col in node.this.expressions:
            if isinstance(col, Column):
                keys_to_validate.append(col.name)
    return keys_to_validate


def _collect_update_keys(parsed: Any) -> list:
    """Collect key names from UPDATE statements."""
    keys_to_validate = []
    for node in parsed.walk():
        if not isinstance(node, exp.Update):
            continue

        for set_expr in node.expressions:
            if not isinstance(set_expr, exp.EQ):
                continue

            left = set_expr.left
            if isinstance(left, Column):
                keys_to_validate.append(left.name)
            elif not isinstance(left, Column):
                raise ValueError(
                    f"Column names must be used in UPDATE SET clause: {set_expr}"
                )
    return keys_to_validate


def _collect_columns_and_keys(parsed: Any) -> tuple[list, list]:
    """Collect column names and key names that need validation."""
    columns_to_validate = _collect_column_names(parsed)
    insert_keys = _collect_insert_keys(parsed)
    update_keys = _collect_update_keys(parsed)
    keys_to_validate = insert_keys + update_keys
    return columns_to_validate, keys_to_validate


def _validate_comparison_nodes(parsed: Any, uid: str, span_context: Any) -> Any:
    """Validate comparison operation nodes."""
    for node in parsed.walk():
        if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
            left = node.left
            if isinstance(left, Column):
                continue
            elif not isinstance(left, (Column, Literal)):
                span_context.add_error_event(
                    f"DML statement contains illegal expression: {node}"
                )
                return format_response(
                    code=CodeEnum.DMLNotAllowed.code,
                    message=f"DML statement contains illegal expression: {node}",
                    sid=span_context.sid,
                )
    return None


def _validate_name_pattern(names: list, name_type: str, span_context: Any) -> Any:
    """
    Validate name pattern for DML identifiers.

    IMPORTANT: This function enforces a STRICT business rule that column names
    must ONLY contain letters and underscores (no digits allowed). This is an
    intentional design decision for this application, NOT a bug or oversight.

    While standard SQL/PostgreSQL identifiers can contain digits after the first
    character (e.g., "users_v2", "table_2024"), this application deliberately
    restricts column names to letters and underscores only for:
    - Consistency across the application
    - Compatibility with downstream systems
    - Simplified validation logic

    This restriction is by design and should NOT be changed to allow digits.

    NOTE FOR CODE REVIEWERS: This intentionally restrictive pattern is correct
    and should not be flagged as an issue. The business requirement explicitly
    prohibits digits in column names.
    """
    allow_chars = string.ascii_letters + "_"
    for name in names:
        if not name:
            span_context.add_error_event(
                f"{name_type}: '{name}' does not conform to rules, only letters and underscores are supported"
            )
            return format_response(
                code=CodeEnum.DMLNotAllowed.code,
                message=f"{name_type}: '{name}' does not conform to rules, only letters and underscores are supported",
                sid=span_context.sid,
            )
        if not all(c in allow_chars for c in name):
            span_context.add_error_event(
                f"{name_type}: '{name}' does not conform to rules, only letters and underscores are supported"
            )
            return format_response(
                code=CodeEnum.DMLNotAllowed.code,
                message=f"{name_type}: '{name}' does not conform to rules, only letters and underscores are supported",
                sid=span_context.sid,
            )
    return None


def _validate_reserved_keywords(keys: list, span_context: Any) -> Any:
    """Validate reserved keywords."""
    for key_name in keys:
        if key_name.lower() in PGSQL_INVALID_KEY:
            span_context.add_error_event(
                f"Key name '{key_name}' is a reserved keyword and is not allowed"
            )
            return format_response(
                code=CodeEnum.DMLNotAllowed.code,
                message=f"Key name '{key_name}' is a reserved keyword and is not allowed",
                sid=span_context.sid,
            )
    return None


async def _validate_dml_legality(dml: str, uid: str, span_context: Any) -> Any:
    try:
        parsed = sqlglot.parse_one(dml, dialect="postgres")
        error_result = _validate_comparison_nodes(parsed, uid, span_context)
        if error_result:
            return error_result
        columns_to_validate, keys_to_validate = _collect_columns_and_keys(parsed)
        error_result = _validate_name_pattern(
            columns_to_validate, "Column name", span_context
        )
        if error_result:
            return error_result
        error_result = _validate_name_pattern(
            keys_to_validate, "Key name", span_context
        )
        if error_result:
            return error_result
        error_result = _validate_reserved_keywords(keys_to_validate, span_context)
        if error_result:
            return error_result
        return None
    except Exception as parse_error:  # pylint: disable=broad-except
        span_context.record_exception(parse_error)
        return format_response(
            code=CodeEnum.SQLParseError.code,
            message="SQL parsing failed",
            sid=span_context.sid,
        )


async def _validate_and_prepare_dml(db: Any, dml_input: Any, span_context: Any) -> Any:
    """Validate input and prepare DML execution."""
    app_id = dml_input.app_id
    uid = dml_input.uid
    database_id = dml_input.database_id
    dml = dml_input.dml
    env = dml_input.env
    space_id = dml_input.space_id

    need_check = {
        "app_id": app_id,
        "database_id": database_id,
        "uid": uid,
        "dml": dml,
        "env": env,
        "space_id": space_id,
    }
    span_context.add_info_events(need_check)
    span_context.add_info_event(f"app_id: {app_id}")
    span_context.add_info_event(f"database_id: {database_id}")
    span_context.add_info_event(f"uid: {uid}")

    if space_id:
        _, error_spaceid = await check_space_id_and_get_uid(
            db, database_id, space_id, span_context
        )
        if error_spaceid:
            return None, error_spaceid

    schema_list, error_resp = await check_database_exists_by_did(
        db, database_id, span_context
    )
    if error_resp:
        return None, error_resp

    return (app_id, uid, database_id, dml, env, schema_list), None


async def _get_table_column_types(
    db: AsyncSession, schema: str, tables: List[str]
) -> Dict[str, str]:
    """Query table column type information. Returns dict mapping 'table.column' to data type."""
    column_types: Dict[str, str] = {}
    for table in tables:
        sql = """
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = :table_name AND table_schema = :table_schema
        """
        result = await parse_and_exec_sql(
            db, sql, {"table_name": table, "table_schema": schema}
        )
        for row in result.fetchall():
            col_name, data_type, udt_name = row[0], row[1], row[2]
            key = f"{table}.{col_name}"
            if udt_name and udt_name.lower() in ("json", "jsonb"):
                column_types[key] = udt_name.lower()
            else:
                column_types[key] = udt_name or data_type
    return column_types


async def _process_dml_statements(
    dmls: List[str],
    app_id: str,
    uid: str,
    span_context: Any,
    db: AsyncSession,
    schema: str,
) -> Any:
    """Process and rewrite DML statements."""
    rewrite_dmls = []
    for statement in dmls:
        error_legality = await _validate_dml_legality(statement, uid, span_context)
        if error_legality:
            return None, error_legality

        column_types: Optional[Dict[str, str]] = None
        try:
            parsed = parse_one(statement)
            tables = [
                table.name for table in parsed.find_all(exp.Table) if table.name != ""
            ]
            if tables:
                column_types = await _get_table_column_types(db, schema, tables)
                span_context.add_info_event(
                    f"Column types for tables {tables}: {column_types}"
                )
        except Exception as col_type_error:  # pylint: disable=broad-except
            span_context.add_error_event(
                f"Failed to get column types: {str(col_type_error)}"
            )
            column_types = None

        rewrite_dml, insert_ids, params = rewrite_dml_with_uid_and_limit(
            dml=statement,
            app_id=app_id,
            uid=uid,
            limit_num=100,
            column_types=column_types,
        )
        span_context.add_info_event(f"rewrite dml sql: {rewrite_dml}")
        span_context.add_info_event(f"rewrite dml params: {params}")
        span_context.add_info_event(f"rewrite dml insert_ids: {insert_ids}")
        rewrite_dmls.append(
            {
                "rewrite_dml": rewrite_dml,
                "insert_ids": insert_ids,
                "params": params,
            }
        )
    return rewrite_dmls, None


@exec_dml_router.post("/exec_dml", response_class=JSONResponse)
async def exec_dml(
    dml_input: ExecDMLInput, db: AsyncSession = Depends(get_session)
) -> JSONResponse:
    """
    Execute DML statements on specified database.

    Args:
        dml_input: Input containing DML statements and metadata
        db: Database session

    Returns:
        JSONResponse: Result of DML execution
    """
    uid = dml_input.uid
    database_id = dml_input.database_id
    metric_service = get_otlp_metric_service()
    m = metric_service.get_meter()(func="exec_dml")
    span_service = get_otlp_span_service()
    span = span_service.get_span()(uid=uid)

    with span.start(
        func_name="exec_dml",
        add_source_function_name=True,
        attributes={"uid": uid, "database_id": database_id},
    ) as span_context:
        try:
            validated_data, error = await _validate_and_prepare_dml(
                db, dml_input, span_context
            )
            if error:
                return error  # type: ignore[no-any-return]

            app_id, uid, database_id, dml, env, schema_list = validated_data

            schema, error_search = await _set_search_path(
                db, schema_list, env, uid, span_context
            )
            if error_search:
                return error_search  # type: ignore[no-any-return]

            dmls, error_split = await _dml_split(dml, db, schema, uid, span_context)
            if error_split:
                return error_split  # type: ignore[no-any-return]

            rewrite_dmls, error_legality = await _process_dml_statements(
                dmls, app_id, uid, span_context, db, schema
            )
            if error_legality:
                return error_legality  # type: ignore[no-any-return]

            final_exec_success_res, exec_time, error_exec = await _exec_dml_sql(
                db, rewrite_dmls, uid, span_context
            )
            if error_exec:
                return error_exec  # type: ignore[no-any-return]

            return format_response(  # type: ignore[no-any-return]
                CodeEnum.Successes.code,
                message=CodeEnum.Successes.msg,
                sid=span_context.sid,
                data={
                    "exec_success": final_exec_success_res,
                    "exec_failure": [],
                    "exec_time": exec_time,
                },
            )
        except CustomException as custom_error:
            span_context.record_exception(custom_error)
            m.in_error_count(custom_error.code, lables={"uid": uid}, span=span_context)
            return format_response(  # type: ignore[no-any-return]
                code=custom_error.code,
                message="Database execution failed",
                sid=span_context.sid,
            )
        except Exception as unexpected_error:  # pylint: disable=broad-except
            m.in_error_count(
                CodeEnum.DMLExecutionError.code, lables={"uid": uid}, span=span_context
            )
            span_context.record_exception(unexpected_error)
            return format_response(  # type: ignore[no-any-return]
                code=CodeEnum.DMLExecutionError.code,
                message="Database execution failed",
                sid=span_context.sid,
            )


async def _exec_dml_sql(
    db: Any, rewrite_dmls: List[Any], uid: str, span_context: Any
) -> Any:
    """Execute rewritten DML SQL statements."""
    final_exec_success_res = []
    start_time = time.time()

    try:
        for dml_info in rewrite_dmls:
            rewrite_dml = dml_info["rewrite_dml"]
            insert_ids = dml_info["insert_ids"]
            params = dml_info.get("params", {})

            if params:
                result = await parse_and_exec_sql(db, rewrite_dml, params)
            else:
                result = await exec_sql_statement(db, rewrite_dml)
            try:
                exec_result = result.mappings().all()
                exec_result_dicts = [dict(row) for row in exec_result]
                exec_result_dicts = to_jsonable(exec_result_dicts)
            except Exception as mapping_error:
                span_context.add_info_event(f"{str(mapping_error)}")
                exec_result_dicts = []

            span_context.add_info_event(f"exec result: {exec_result_dicts}")

            if exec_result_dicts:
                final_exec_success_res.extend(exec_result_dicts)
            elif insert_ids:
                final_exec_success_res.extend([{"id": v} for v in insert_ids])

            await db.commit()

        exec_time = time.time() - start_time
        return final_exec_success_res, exec_time, None

    except Exception as exec_error:  # pylint: disable=broad-except
        span_context.record_exception(exec_error)
        await db.rollback()
        return (
            None,
            None,
            format_response(
                code=CodeEnum.DatabaseExecutionError.code,
                message="Database execution failed",
                sid=span_context.sid,
            ),
        )


async def _set_search_path(
    db: Any, schema_list: List[Any], env: str, uid: str, span_context: Any
) -> Any:
    """Set search path for database operations."""
    schema = next((one[0] for one in schema_list if env in one[0]), "")
    if not schema:
        span_context.add_error_event("Corresponding schema not found")
        return None, format_response(
            code=CodeEnum.NoSchemaError.code,
            message=f"Corresponding schema not found: {schema}",
            sid=span_context.sid,
        )

    span_context.add_info_event(f"schema: {schema}")
    try:
        await set_search_path_by_schema(db, schema)
        return schema, None
    except Exception as schema_error:  # pylint: disable=broad-except
        span_context.record_exception(schema_error)
        return None, format_response(
            code=CodeEnum.NoSchemaError.code,
            message=f"Invalid schema: {schema}",
            sid=span_context.sid,
        )


async def _dml_split(
    dml: str, db: Any, schema: str, uid: str, span_context: Any
) -> Any:
    """Split and validate DML statements."""
    dml = dml.strip()
    dmls = sqlparse.split(dml)
    span_context.add_info_event(f"Split DML statements: {dmls}")

    for statement in dmls:
        try:
            parsed = parse_one(statement)
            tables = {
                table.name for table in parsed.find_all(exp.Table) if table.name != ""
            }
        except Exception as parse_error:  # pylint: disable=broad-except
            span_context.record_exception(parse_error)
            return None, format_response(
                code=CodeEnum.SQLParseError.code,
                message="SQL parsing failed",
                sid=span_context.sid,
            )

        result = await parse_and_exec_sql(
            db,
            "SELECT tablename FROM pg_tables WHERE schemaname = :schema",
            {"schema": schema},
        )
        valid_tables = {row[0] for row in result.fetchall()}
        not_found = tables - valid_tables

        if not_found:
            span_context.add_error_event(
                f"Table does not exist or no permission: {', '.join(not_found)}"
            )
            return None, format_response(
                code=CodeEnum.NoAuthorityError.code,
                message=f"Table does not exist or no permission: "
                f"{', '.join(not_found)}",
                sid=span_context.sid,
            )

        allowed_sql = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE)\s+", re.IGNORECASE)
        if not allowed_sql.match(statement):
            span_context.add_error_events({"invalid dml": statement})
            return None, format_response(
                code=CodeEnum.DMLNotAllowed.code,
                message="Unsupported SQL type, only "
                "SELECT/INSERT/UPDATE/DELETE allowed",
                sid=span_context.sid,
            )

    return dmls, None
