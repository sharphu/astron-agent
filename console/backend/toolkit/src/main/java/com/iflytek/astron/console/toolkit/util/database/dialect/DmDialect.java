package com.iflytek.astron.console.toolkit.util.database.dialect;

import com.iflytek.astron.console.toolkit.util.database.SqlRenderer;
import org.apache.commons.lang3.StringUtils;
import org.jooq.SQLDialect;

import java.util.List;

/**
 * Dameng DM dialect implementation.
 *
 * Key goals for integration:
 * - Use double-quote identifiers (matches user requirement).
 * - Use DM8 native IDENTITY(1,1) for auto-increment.
 * - Use inline column COMMENT in column definitions.
 *
 * Note: For robustness with the memory/database SQLGlot parsing pipeline,
 * we intentionally keep table-level comment generation minimal.
 */
public class DmDialect implements DbDialect {

    @Override
    public String quoteIdent(String name) {
        String n = SqlRenderer.validateIdent(name);
        return "\"" + n.replace("\"", "\"\"") + "\"";
    }

    @Override
    public String buildCreateTable(String tableName, List<ColumnDef> columns, String tableComment) {
        StringBuilder ddl = new StringBuilder();
        String table = quoteIdent(tableName);

        // System columns: id/uid/create_time.
        ddl.append("CREATE TABLE ").append(table).append(" (\n")
                .append("  ").append(quoteIdent("id")).append(" BIGINT IDENTITY(1, 1) PRIMARY KEY COMMENT 'Primary key id',\n")
                .append("  ").append(quoteIdent("uid")).append(" VARCHAR(64) NOT NULL COMMENT 'uid',\n")
                .append("  ").append(quoteIdent("create_time")).append(" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP");

        // User columns.
        for (ColumnDef col : columns) {
            ddl.append(",\n  ").append(quoteIdent(col.name())).append(" ").append(col.sqlType());
            if (col.notNull()) {
                ddl.append(" NOT NULL");
            }
            if (col.defaultValue() != null) {
                ddl.append(" DEFAULT ").append(col.defaultValue());
            }
            if (StringUtils.isNotBlank(col.comment())) {
                ddl.append(" COMMENT ").append(SqlRenderer.quoteLiteral(col.comment()));
            }
        }
        ddl.append("\n);");

        // For DM integration safety we omit table-level comment here.
        // (The service-side SQLGlot dialect parsing is not guaranteed for COMMENT ON TABLE.)
        return ddl.toString();
    }

    @Override
    public String buildAddColumn(String tableName, ColumnDef column) {
        StringBuilder sql = new StringBuilder();
        String table = quoteIdent(tableName);
        String col = quoteIdent(column.name());

        sql.append("ALTER TABLE ").append(table)
                .append(" ADD COLUMN ").append(col).append(" ").append(column.sqlType());
        if (column.notNull()) {
            sql.append(" NOT NULL");
        }
        if (column.defaultValue() != null) {
            sql.append(" DEFAULT ").append(column.defaultValue());
        }
        if (StringUtils.isNotBlank(column.comment())) {
            sql.append(" COMMENT ").append(SqlRenderer.quoteLiteral(column.comment()));
        }
        sql.append(";");
        return sql.toString();
    }

    @Override
    public String buildDropColumn(String tableName, String columnName) {
        String table = quoteIdent(tableName);
        String col = quoteIdent(columnName);
        // DM8 syntax: ALTER TABLE 表名 DROP COLUMN 字段名;
        return "ALTER TABLE " + table + " DROP COLUMN " + col + ";";
    }

    @Override
    public String buildModifyColumn(String tableName, ColumnModification mod) {
        StringBuilder sql = new StringBuilder();
        String table = quoteIdent(tableName);

        // Rename first (when needed).
        if (!mod.oldName().equals(mod.newName())) {
            sql.append("ALTER TABLE ").append(table)
                    .append(" RENAME COLUMN ").append(quoteIdent(mod.oldName()))
                    .append(" TO ").append(quoteIdent(mod.newName()))
                    .append(";");
        }

        // Modify the (possibly renamed) column definition.
        sql.append("ALTER TABLE ").append(table)
                .append(" MODIFY ").append(quoteIdent(mod.newName())).append(" ")
                .append(mod.newSqlType());
        if (mod.newNotNull()) {
            sql.append(" NOT NULL");
        }
        if (mod.newDefault() != null) {
            sql.append(" DEFAULT ").append(mod.newDefault());
        }
        if (StringUtils.isNotBlank(mod.newComment())) {
            sql.append(" COMMENT ").append(SqlRenderer.quoteLiteral(mod.newComment()));
        }
        sql.append(";");

        return sql.toString();
    }

    @Override
    public String buildRenameTable(String oldName, String newName) {
        return "ALTER TABLE " + quoteIdent(oldName) + " RENAME TO " + quoteIdent(newName) + ";";
    }

    @Override
    public String buildTableComment(String tableName, String comment) {
        // Intentionally omit table-level comment for DM to avoid SQLGlot parsing incompatibilities.
        // Column comments are generated inline during create/add/modify.
        return "";
    }

    @Override
    public SQLDialect jooqDialect() {
        // jOOQ only needs an SQL dialect constant for some utility SQL generation.
        // We choose POSTGRES as the closest ANSI baseline here.
        return SQLDialect.POSTGRES;
    }
}
