-- Migration script for init_core

DROP TABLE IF EXISTS agent_apply_record;
CREATE TABLE agent_apply_record
(
    id             BIGSERIAL,
    enterprise_id  bigint       DEFAULT NULL ,
    space_id       bigint       DEFAULT NULL ,
    apply_uid      varchar(128) DEFAULT NULL ,
    apply_nickname varchar(64)  DEFAULT NULL ,
    apply_time     TIMESTAMP     DEFAULT NULL ,
    status         SMALLINT      DEFAULT NULL ,
    audit_time     TIMESTAMP     DEFAULT NULL ,
    audit_uid      varchar(128) DEFAULT NULL ,
    create_time    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX agent_apply_record_enterprise_id_key ON agent_apply_record (enterprise_id);
CREATE INDEX agent_apply_record_space_id_key ON agent_apply_record (space_id);
CREATE INDEX agent_apply_record_apply_uid_key ON agent_apply_record (apply_uid);

DROP TABLE IF EXISTS agent_invite_record;
CREATE TABLE agent_invite_record
(
    id               BIGSERIAL,
    type             SMALLINT      DEFAULT NULL ,
    space_id         bigint       DEFAULT NULL ,
    enterprise_id    bigint       DEFAULT NULL ,
    invitee_uid      varchar(128) DEFAULT NULL ,
    role             SMALLINT      DEFAULT NULL ,
    invitee_nickname varchar(64)  DEFAULT NULL ,
    inviter_uid      varchar(128) DEFAULT NULL ,
    expire_time      TIMESTAMP     DEFAULT NULL ,
    status           SMALLINT      DEFAULT NULL ,
    create_time      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time      TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX agent_invite_record_invitee_id_key ON agent_invite_record (invitee_uid);
CREATE INDEX agent_invite_record_space_id_key ON agent_invite_record (space_id);
CREATE INDEX agent_invite_record_enterprise_id_key ON agent_invite_record (enterprise_id);

DROP TABLE IF EXISTS agent_share_record;
CREATE TABLE agent_share_record
(
    id          BIGSERIAL,
    uid         varchar(128) NOT NULL ,
    base_id     bigint       NOT NULL ,
    share_key   varchar(64) DEFAULT '' ,
    share_type  SMALLINT     DEFAULT '0' ,
    is_act      SMALLINT     DEFAULT '1' ,
    create_time TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX agent_share_record_idx_uid ON agent_share_record (uid);
CREATE INDEX agent_share_record_idx_base_id ON agent_share_record (base_id);
CREATE INDEX agent_share_record_idx_share_key ON agent_share_record (share_key);

DROP TABLE IF EXISTS ai_prompt_template;
CREATE TABLE ai_prompt_template
(
    id             BIGSERIAL ,
    prompt_key     varchar(100) NOT NULL ,
    language_code  varchar(10)  NOT NULL ,
    prompt_content text NOT NULL ,
    is_active      SMALLINT DEFAULT '1' ,
    created_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ,
    updated_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX ai_prompt_template_uk_prompt_key_lang ON ai_prompt_template (prompt_key,language_code);
CREATE INDEX ai_prompt_template_idx_is_active ON ai_prompt_template (is_active);

DROP TABLE IF EXISTS application_form;
CREATE TABLE application_form
(
    id          BIGSERIAL ,
    nickname    varchar(255) NOT NULL ,
    mobile      varchar(255) NOT NULL ,
    bot_name    varchar(255) NOT NULL ,
    bot_id      bigint       NOT NULL ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX application_form_idx_bot_id ON application_form (bot_id);

DROP TABLE IF EXISTS auth_apply_record;
CREATE TABLE auth_apply_record
(
    id            SERIAL,
    app_id        varchar(128) DEFAULT NULL,
    domain        varchar(255) DEFAULT NULL,
    content       text,
    create_time   TIMESTAMP     DEFAULT NULL,
    uid           varchar(128) DEFAULT NULL,
    channel       varchar(255) DEFAULT NULL,
    patch_id      varchar(128) DEFAULT NULL,
    auto_auth     BOOLEAN       DEFAULT NULL,
    auth_order_id varchar(255) DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS call_log;
CREATE TABLE call_log
(
    id          BIGSERIAL,
    sid         varchar(255) DEFAULT NULL,
    req         text,
    resp        text,
    create_time TIMESTAMP     DEFAULT NULL,
    type        varchar(255) DEFAULT NULL,
    url         varchar(512) DEFAULT NULL,
    method      varchar(64)  DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS chat_info;
CREATE TABLE chat_info
(
    id              BIGSERIAL,
    app_id          varchar(255) DEFAULT NULL,
    bot_id          varchar(255) DEFAULT NULL,
    flow_id         varchar(255) DEFAULT NULL,
    sub             varchar(255) DEFAULT NULL ,
    caller          varchar(255) DEFAULT NULL ,
    log_caller      varchar(32)  DEFAULT '',
    uid             varchar(255) DEFAULT NULL,
    sid             varchar(255) DEFAULT NULL,
    question        text,
    answer          text,
    status_code     int          DEFAULT NULL,
    message         text ,
    total_cost_time int          DEFAULT NULL ,
    first_cost_time int          DEFAULT NULL ,
    token           int          DEFAULT NULL ,
    create_time     TIMESTAMP     DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX chat_info_app_id ON chat_info (app_id);
CREATE INDEX chat_info_bot_id ON chat_info (bot_id);
CREATE INDEX chat_info_sid ON chat_info (sid);
CREATE INDEX chat_info_index_6 ON chat_info (flow_id);
CREATE INDEX chat_info_log_caller ON chat_info (log_caller);
CREATE INDEX chat_info_status_code ON chat_info (status_code);
CREATE INDEX chat_info_bot_id_IDX ON chat_info (bot_id,sub,caller,create_time);
CREATE INDEX chat_info_idx_sub_create_time ON chat_info (sub,create_time);

DROP TABLE IF EXISTS chat_list;
CREATE TABLE chat_list
(
    id                 BIGSERIAL ,
    uid                varchar(128)      DEFAULT NULL ,
    title              varchar(255)      DEFAULT NULL ,
    is_delete          SMALLINT           DEFAULT '0' ,
    enable             SMALLINT           DEFAULT '1' ,
    bot_id             int               DEFAULT '0' ,
    sticky             SMALLINT  NOT NULL DEFAULT '0' ,
    create_time        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time        TIMESTAMP          DEFAULT CURRENT_TIMESTAMP ,
    is_model           SMALLINT  NOT NULL DEFAULT '0' ,
    enabled_plugin_ids varchar(255)      DEFAULT '' ,
    is_botweb          SMALLINT  NOT NULL DEFAULT '0' ,
    file_id            varchar(64)       DEFAULT NULL ,
    root_flag          SMALLINT  NOT NULL DEFAULT '1' ,
    personality_id     bigint            DEFAULT '0' ,
    gcl_id             bigint            DEFAULT '0' ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_list_create_time_IDX ON chat_list (create_time);
CREATE INDEX chat_list_idx_bot_id ON chat_list (bot_id);
CREATE INDEX chat_list_idx_uid_bid_ctime ON chat_list (uid,bot_id,create_time);
CREATE INDEX chat_list_file_id_idx ON chat_list (file_id);
CREATE INDEX chat_list_idx_pid_uid ON chat_list (personality_id,uid);

DROP TABLE IF EXISTS chat_reanwser_records;
CREATE TABLE chat_reanwser_records
(
    id          BIGSERIAL ,
    uid         varchar(128)      DEFAULT NULL ,
    chat_id     bigint            DEFAULT NULL ,
    req_id      bigint            DEFAULT NULL ,
    ask         varchar(8000)     DEFAULT NULL ,
    answer      varchar(8000)     DEFAULT NULL ,
    ask_time    TIMESTAMP          DEFAULT NULL ,
    answer_time TIMESTAMP          DEFAULT NULL ,
    sid         varchar(64)       DEFAULT NULL ,
    answer_type SMALLINT           DEFAULT NULL ,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP          DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_reanwser_records_uid_index ON chat_reanwser_records (uid);
CREATE INDEX chat_reanwser_records_chat_index ON chat_reanwser_records (chat_id);
CREATE INDEX chat_reanwser_records_idx_sid ON chat_reanwser_records (sid);

DROP TABLE IF EXISTS chat_reason_records;
CREATE TABLE chat_reason_records
(
    id                    BIGSERIAL,
    uid                   varchar(128) NOT NULL ,
    chat_id               bigint       NOT NULL ,
    req_id                bigint       NOT NULL ,
    content               TEXT     NOT NULL ,
    thinking_elapsed_secs bigint                DEFAULT '0' ,
    type                  varchar(50)           DEFAULT NULL ,
    create_time           TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time           TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_reason_records_idx_uid ON chat_reason_records (uid);
CREATE INDEX chat_reason_records_idx_chat_id ON chat_reason_records (chat_id);
CREATE INDEX chat_reason_records_idx_req_id ON chat_reason_records (req_id);
CREATE INDEX chat_reason_records_idx_create_time ON chat_reason_records (create_time);

DROP TABLE IF EXISTS chat_req_records;
CREATE TABLE chat_req_records
(
    id          BIGSERIAL,
    chat_id     bigint   NOT NULL ,
    uid         varchar(128)      DEFAULT NULL ,
    message     varchar(8000)     DEFAULT NULL ,
    client_type SMALLINT           DEFAULT '0' ,
    model_id    int               DEFAULT NULL ,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP          DEFAULT CURRENT_TIMESTAMP ,
    date_stamp  int               DEFAULT NULL ,
    new_context SMALLINT  NOT NULL DEFAULT '1' ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_req_records_idx_chat_id ON chat_req_records (chat_id);
CREATE INDEX chat_req_records_idx_create_time ON chat_req_records (create_time);
CREATE INDEX chat_req_records_idx_date_stamp ON chat_req_records (date_stamp);
CREATE INDEX chat_req_records_idx_uid_chatId ON chat_req_records (uid,chat_id);

DROP TABLE IF EXISTS chat_resp_alltool_data;
CREATE TABLE chat_resp_alltool_data
(
    id          BIGSERIAL,
    uid         varchar(128) DEFAULT NULL ,
    chat_id     bigint       DEFAULT NULL ,
    req_id      bigint       DEFAULT NULL ,
    seq_no      varchar(100) DEFAULT NULL ,
    tool_data   text ,
    tool_name   varchar(100) DEFAULT NULL ,
    create_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX chat_resp_alltool_data_uid_IDX ON chat_resp_alltool_data (uid);
CREATE INDEX chat_resp_alltool_data_chat_id_IDX ON chat_resp_alltool_data (chat_id);
CREATE INDEX chat_resp_alltool_data_req_id_IDX ON chat_resp_alltool_data (req_id);

DROP TABLE IF EXISTS chat_resp_records;
CREATE TABLE chat_resp_records
(
    id          BIGSERIAL,
    uid         varchar(128)                                                  DEFAULT NULL ,
    chat_id     bigint                                                        DEFAULT NULL ,
    req_id      bigint                                                        DEFAULT NULL ,
    sid         varchar(128) DEFAULT NULL ,
    answer_type SMALLINT                                                       DEFAULT '2' ,
    message     TEXT ,
    create_time TIMESTAMP NOT NULL                                             DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP                                                      DEFAULT CURRENT_TIMESTAMP ,
    date_stamp  int                                                           DEFAULT NULL ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_resp_records_idx_chat_id ON chat_resp_records (chat_id);
CREATE INDEX chat_resp_records_idx_create_time ON chat_resp_records (create_time);
CREATE INDEX chat_resp_records_idx_reqId ON chat_resp_records (req_id);
CREATE INDEX chat_resp_records_idx_sid ON chat_resp_records (sid);
CREATE INDEX chat_resp_records_idx_uid_chatId ON chat_resp_records (uid,chat_id);

DROP TABLE IF EXISTS chat_token_records;
CREATE TABLE chat_token_records
(
    id                BIGSERIAL,
    sid               varchar(64) DEFAULT NULL ,
    prompt_tokens     int         DEFAULT NULL ,
    question_tokens   int         DEFAULT NULL ,
    completion_tokens int         DEFAULT NULL ,
    total_tokens      int         DEFAULT NULL ,
    create_time       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    update_time       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX chat_token_records_idx_create_time ON chat_token_records (create_time);
CREATE INDEX chat_token_records_idx_sid ON chat_token_records (sid);

DROP TABLE IF EXISTS chat_trace_source;
CREATE TABLE chat_trace_source
(
    id          BIGSERIAL,
    uid         varchar(128) DEFAULT NULL ,
    chat_id     bigint       DEFAULT NULL ,
    req_id      bigint       DEFAULT NULL ,
    content     text ,
    type        varchar(50)  DEFAULT 'search' ,
    create_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX chat_trace_source_chat_id_IDX ON chat_trace_source (chat_id);
CREATE INDEX chat_trace_source_type_IDX ON chat_trace_source (type);
CREATE INDEX chat_trace_source_uid_IDX ON chat_trace_source (uid);

DROP TABLE IF EXISTS chat_tree_index;
CREATE TABLE chat_tree_index
(
    id             BIGSERIAL,
    root_chat_id   bigint   NOT NULL ,
    parent_chat_id bigint   NOT NULL ,
    child_chat_id  bigint   NOT NULL ,
    uid            varchar(128)      DEFAULT NULL ,
    create_time    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_tree_index_uid_IDX ON chat_tree_index (uid);
CREATE INDEX chat_tree_index_root_chat_id_IDX ON chat_tree_index (root_chat_id);
CREATE INDEX chat_tree_index_idx_child_chat_id ON chat_tree_index (child_chat_id);

DROP TABLE IF EXISTS chat_user;
CREATE TABLE chat_user
(
    id             BIGSERIAL ,
    uid            varchar(128) DEFAULT NULL ,
    name           varchar(255) DEFAULT NULL ,
    avatar         varchar(512) DEFAULT NULL ,
    nickname       varchar(255) DEFAULT NULL ,
    mobile         varchar(255) NOT NULL ,
    is_able        SMALLINT      DEFAULT '0' ,
    create_time    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    user_agreement SMALLINT      DEFAULT '0' ,
    date_stamp     int          DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX chat_user_uid_unique_index ON chat_user (uid);
CREATE INDEX chat_user_idx_create_time ON chat_user (create_time);
CREATE INDEX chat_user_index_mobile ON chat_user (mobile);
CREATE INDEX chat_user_idx_nickname ON chat_user (nickname);

DROP TABLE IF EXISTS config_info;
CREATE TABLE config_info
(
    id          BIGSERIAL ,
    category    varchar(64)   DEFAULT NULL ,
    code        varchar(128)  DEFAULT NULL ,
    name        varchar(255)  DEFAULT NULL ,
    value       text ,
    is_valid    SMALLINT       DEFAULT NULL ,
    remarks     varchar(1000) DEFAULT NULL ,
    create_time TIMESTAMP      DEFAULT '2000-01-01 00:00:00' ,
    update_time TIMESTAMP      DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS config_info_en;
CREATE TABLE config_info_en
(
    id          BIGSERIAL ,
    category    varchar(64)   DEFAULT NULL ,
    code        varchar(128)  DEFAULT NULL ,
    name        varchar(255)  DEFAULT NULL ,
    value       text ,
    is_valid    SMALLINT NOT NULL ,
    remarks     varchar(1000) DEFAULT NULL ,
    create_time TIMESTAMP      DEFAULT '2000-01-01 00:00:00' ,
    update_time TIMESTAMP      DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS core_system_error_code;
CREATE TABLE core_system_error_code
(
    id         SERIAL,
    error_code int          NOT NULL,
    error_msg  varchar(100) NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS custom_vcn;
CREATE TABLE custom_vcn
(
    id          BIGSERIAL,
    uid         bigint                                                        DEFAULT NULL,
    name        varchar(64)  DEFAULT NULL,
    status      SMALLINT                                                       DEFAULT NULL ,
    vcn_code    varchar(255) DEFAULT NULL ,
    try_vcn_url varchar(255) DEFAULT NULL ,
    task_id     bigint                                                        DEFAULT NULL ,
    vcn_task_id varchar(255) DEFAULT NULL ,
    sex         SMALLINT                                                       DEFAULT NULL,
    create_time TIMESTAMP                                                      DEFAULT NULL,
    update_time TIMESTAMP                                                      DEFAULT NULL,
    share       SMALLINT                                                       DEFAULT '0' ,
    agent_id    bigint                                                        DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX custom_vcn_idx_agent_id ON custom_vcn (agent_id);
CREATE INDEX custom_vcn_idx_task_id ON custom_vcn (task_id);
CREATE INDEX custom_vcn_idx_uid ON custom_vcn (uid);
CREATE INDEX custom_vcn_idx_vcn_code ON custom_vcn (vcn_code);
CREATE INDEX custom_vcn_idx_vcn_task_id ON custom_vcn (vcn_task_id);

DROP TABLE IF EXISTS db_info;
CREATE TABLE db_info
(
    id           BIGSERIAL,
    app_id       varchar(100) NOT NULL,
    uid          varchar(100) NOT NULL ,
    db_id        bigint                DEFAULT NULL ,
    name         varchar(100) NOT NULL ,
    description  varchar(255)          DEFAULT NULL ,
    avatar_icon  varchar(255)          DEFAULT NULL ,
    avatar_color varchar(255)          DEFAULT NULL,
    deleted      SMALLINT      NOT NULL DEFAULT '0',
    create_time  TIMESTAMP     NOT NULL,
    update_time  TIMESTAMP              DEFAULT NULL,
    space_id     bigint                DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS db_table;
CREATE TABLE db_table
(
    id          BIGSERIAL,
    db_id       bigint       NOT NULL ,
    name        varchar(100) NOT NULL,
    description varchar(255)          DEFAULT NULL,
    deleted     SMALLINT      NOT NULL DEFAULT '0',
    create_time TIMESTAMP     NOT NULL,
    update_time TIMESTAMP              DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS db_table_field;
CREATE TABLE db_table_field
(
    id            BIGSERIAL,
    tb_id         bigint       NOT NULL ,
    name          varchar(100) NOT NULL,
    type          varchar(100) NOT NULL,
    description   varchar(100)          DEFAULT NULL,
    default_value varchar(100)          DEFAULT NULL,
    is_required   SMALLINT      NOT NULL DEFAULT '0',
    is_system     SMALLINT      NOT NULL DEFAULT '0',
    create_time   TIMESTAMP     NOT NULL,
    update_time   TIMESTAMP     NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS exclude_appid_flowId;
CREATE TABLE exclude_appid_flowId
(
    id      SERIAL,
    app_id  varchar(100) DEFAULT NULL,
    flow_id varchar(100) DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX exclude_appid_flowId_app_id_IDX ON exclude_appid_flowId (app_id);
CREATE INDEX exclude_appid_flowId_flow_id_IDX ON exclude_appid_flowId (flow_id);

DROP TABLE IF EXISTS feedback_info;
CREATE TABLE feedback_info
(
    id          BIGSERIAL,
    app_id      varchar(255)  DEFAULT NULL,
    sub         varchar(255)  DEFAULT NULL,
    uid         varchar(128)  DEFAULT NULL,
    chat_id     varchar(128)  DEFAULT NULL,
    sid         varchar(128)  DEFAULT NULL,
    bot_id      varchar(128)  DEFAULT NULL,
    flow_id     varchar(128)  DEFAULT NULL,
    question    text,
    answer      text,
    action      varchar(255)  DEFAULT NULL,
    reason      varchar(255)  DEFAULT NULL,
    remark      varchar(1200) DEFAULT NULL,
    create_time TIMESTAMP      DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX feedback_info_app_id ON feedback_info (app_id);
CREATE INDEX feedback_info_uid ON feedback_info (uid);
CREATE INDEX feedback_info_sid ON feedback_info (sid);
CREATE INDEX feedback_info_bot_id ON feedback_info (bot_id);
CREATE INDEX feedback_info_flow_id ON feedback_info (flow_id);

DROP TABLE IF EXISTS fine_tune_task;
CREATE TABLE fine_tune_task
(
    id                    BIGSERIAL,
    optimize_task_id      bigint   NOT NULL,
    dataset_id            bigint   NOT NULL,
    model_id              bigint   NOT NULL,
    fine_tune_task_id     bigint   NOT NULL,
    fine_tune_task_remark varchar(1024) DEFAULT NULL,
    create_time           TIMESTAMP NOT NULL,
    update_time           TIMESTAMP NOT NULL,
    base_model_id         bigint        DEFAULT NULL,
    server_name           varchar(255)  DEFAULT NULL,
    optimize_node         text,
    status                SMALLINT       DEFAULT '1',
    server_id             bigint        DEFAULT NULL,
    server_status         SMALLINT       DEFAULT '0',
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS group_tag;
CREATE TABLE group_tag
(
    id          BIGSERIAL,
    uid         varchar(128)       DEFAULT NULL ,
    name        varchar(64)        DEFAULT NULL ,
    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS group_user;
CREATE TABLE group_user
(
    id          BIGSERIAL,
    uid         varchar(128)       DEFAULT NULL ,
    user_id     varchar(128)       DEFAULT NULL ,
    tag_id      bigint             DEFAULT NULL ,
    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS group_visibility;
CREATE TABLE group_visibility
(
    id          BIGSERIAL,
    uid         varchar(128)       DEFAULT NULL,
    type        int                DEFAULT NULL ,
    user_id     varchar(128)       DEFAULT NULL,
    relation_id varchar(200)       DEFAULT NULL ,
    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    space_id    bigint             DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX group_visibility_type_rel_idx ON group_visibility (type,relation_id);

DROP TABLE IF EXISTS hit_test_history;
CREATE TABLE hit_test_history
(
    id          BIGSERIAL,
    user_id     varchar(128) NOT NULL DEFAULT '-999' ,
    repo_id     bigint       NOT NULL ,
    query       text         NOT NULL ,
    create_time timestamp NULL DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS maas_template;
CREATE TABLE maas_template
(
    id             BIGSERIAL,
    core_abilities JSONB                                                           DEFAULT NULL,
    core_scenarios JSONB                                                           DEFAULT NULL,
    is_act         SMALLINT                                                        DEFAULT NULL,
    maas_id        bigint                                                         DEFAULT NULL,
    subtitle       varchar(128)  DEFAULT NULL,
    title          varchar(128)  DEFAULT NULL,
    bot_id         int                                                            DEFAULT NULL,
    cover_url      varchar(2048) DEFAULT NULL,
    group_id       bigint                                                         DEFAULT NULL,
    order_index    int                                                            DEFAULT NULL,
    create_time    TIMESTAMP NOT NULL                                              DEFAULT CURRENT_TIMESTAMP,
    update_time    TIMESTAMP NOT NULL                                              DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS mcp_data;
CREATE TABLE mcp_data
(
    id           BIGSERIAL ,
    bot_id       bigint                                                        NOT NULL ,
    uid          bigint                                                        NOT NULL ,
    space_id     bigint                                                                 DEFAULT NULL ,
    server_name  varchar(255) NOT NULL ,
    description  text ,
    content      TEXT ,
    icon         varchar(1024)         DEFAULT NULL ,
    server_url   varchar(1024)         DEFAULT NULL ,
    args         JSONB                                                                   DEFAULT NULL ,
    version_name varchar(100)          DEFAULT NULL ,
    released     SMALLINT                                                       NOT NULL DEFAULT '1' ,
    is_delete    SMALLINT                                                       NOT NULL DEFAULT '0' ,
    create_time  TIMESTAMP                                                      NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time  TIMESTAMP                                                      NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX mcp_data_uk_bot_id_version ON mcp_data (bot_id,version_name);
CREATE INDEX mcp_data_idx_uid ON mcp_data (uid);
CREATE INDEX mcp_data_idx_space_id ON mcp_data (space_id);
CREATE INDEX mcp_data_idx_bot_id ON mcp_data (bot_id);
CREATE INDEX mcp_data_idx_released ON mcp_data (released);
CREATE INDEX mcp_data_idx_create_time ON mcp_data (create_time);

DROP TABLE IF EXISTS mcp_tool_config;
CREATE TABLE mcp_tool_config
(
    id          BIGSERIAL,
    mcp_id      varchar(255)          DEFAULT NULL ,
    server_id   varchar(255)          DEFAULT NULL ,
    sort_link   varchar(1024)         DEFAULT NULL ,
    uid         varchar(128) NOT NULL ,
    type        varchar(255)          DEFAULT NULL ,
    content     text ,
    is_deleted  BOOLEAN       NOT NULL DEFAULT FALSE ,
    create_time TIMESTAMP              DEFAULT NULL,
    update_time TIMESTAMP              DEFAULT NULL,
    parameters  text ,
    customize   BOOLEAN                DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS node_info;
CREATE TABLE node_info
(
    id                   BIGSERIAL,
    app_id               varchar(255) DEFAULT NULL,
    bot_id               varchar(255) DEFAULT NULL,
    flow_id              varchar(255) DEFAULT NULL,
    sub                  varchar(255) DEFAULT NULL,
    caller               varchar(255) DEFAULT NULL,
    sid                  varchar(255) DEFAULT NULL,
    node_id              varchar(255) DEFAULT NULL,
    node_name            varchar(255) DEFAULT NULL,
    node_type            varchar(255) DEFAULT NULL,
    running_status       BOOLEAN       DEFAULT NULL ,
    node_input           text ,
    node_output          text ,
    config               text ,
    llm_output           text ,
    domain               varchar(255) DEFAULT NULL,
    cost_time            int          DEFAULT NULL ,
    first_cost_time      int          DEFAULT NULL ,
    node_first_cost_time float        DEFAULT NULL ,
    next_log_ids         text ,
    token                int          DEFAULT NULL ,
    create_time          TIMESTAMP     DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX node_info_app_id ON node_info (app_id);
CREATE INDEX node_info_bot_id ON node_info (bot_id);
CREATE INDEX node_info_flow_id ON node_info (flow_id);
CREATE INDEX node_info_sid ON node_info (sid);
CREATE INDEX node_info_node_id ON node_info (node_id);
CREATE INDEX node_info_domain ON node_info (domain);
CREATE INDEX node_info_create_time ON node_info (create_time);
CREATE INDEX node_info_token ON node_info (token);

DROP TABLE IF EXISTS notifications;
CREATE TABLE notifications
(
    id            BIGINT NOT NULL  ,
    type          varchar(16)  NOT NULL ,
    title         varchar(255) NOT NULL ,
    body          text ,
    template_code varchar(64)  DEFAULT NULL ,
    payload       JSONB                                                          DEFAULT NULL ,
    creator_uid   varchar(128) DEFAULT NULL ,
    created_at    TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    expire_at     TIMESTAMP(3) DEFAULT NULL ,
    meta          JSONB                                                          DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX notifications_idx_type_created ON notifications (type,created_at DESC);
CREATE INDEX notifications_idx_expire ON notifications (expire_at);
CREATE INDEX notifications_idx_creator ON notifications (creator_uid);

DROP TABLE IF EXISTS prompt_template;
CREATE TABLE prompt_template
(
    id               BIGSERIAL ,
    uid              varchar(128) NOT NULL ,
    name             varchar(255)          DEFAULT NULL ,
    description      text ,
    deleted          BOOLEAN       NOT NULL DEFAULT FALSE ,
    prompt           text ,
    created_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_time     TIMESTAMP              DEFAULT CURRENT_TIMESTAMP,
    node_category    int                   DEFAULT NULL ,
    adaptation_model text ,
    max_loop_count   bigint                DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS prompt_template_en;
CREATE TABLE prompt_template_en
(
    id               BIGSERIAL ,
    uid              varchar(128) NOT NULL ,
    name             varchar(255)          DEFAULT NULL ,
    description      text ,
    deleted          BOOLEAN       NOT NULL DEFAULT FALSE ,
    prompt           text ,
    created_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_time     TIMESTAMP              DEFAULT CURRENT_TIMESTAMP,
    node_category    int                   DEFAULT NULL ,
    adaptation_model text ,
    max_loop_count   bigint                DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS rpa_info;
CREATE TABLE rpa_info (
  id BIGSERIAL ,
  category varchar(64) DEFAULT NULL ,
  name varchar(255) DEFAULT NULL ,
  value text ,
  is_deleted SMALLINT DEFAULT '0' ,
  remarks varchar(1000) DEFAULT NULL ,
  icon varchar(150) DEFAULT NULL,
  create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ,
  update_time TIMESTAMP DEFAULT NULL ,
  path varchar(100) DEFAULT NULL ,
  PRIMARY KEY (id)
);

DROP TABLE IF EXISTS rpa_user_assistant;
CREATE TABLE rpa_user_assistant
(
    id             BIGSERIAL ,
    user_id        varchar(64) NOT NULL ,
    platform_id    bigint                                                       NOT NULL ,
    assistant_name varchar(128)                                                 NOT NULL ,
    status         SMALLINT                                                      NOT NULL DEFAULT '1' ,
    remarks        varchar(1000)                                                         DEFAULT NULL ,
    icon           varchar(100)                                                          DEFAULT NULL,
    robot_count    int                                                                   DEFAULT NULL,
    space_id       bigint                                                                DEFAULT NULL,
    create_time    TIMESTAMP                                                     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time    TIMESTAMP                                                     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id),
    CONSTRAINT fk_rpa_platform FOREIGN KEY (platform_id) REFERENCES rpa_info (id));

CREATE UNIQUE INDEX rpa_user_assistant_uk_user_assistant_name ON rpa_user_assistant (user_id,assistant_name);
CREATE INDEX rpa_user_assistant_idx_user ON rpa_user_assistant (user_id);
CREATE INDEX rpa_user_assistant_fk_rpa_platform ON rpa_user_assistant (platform_id);

DROP TABLE IF EXISTS rpa_user_assistant_field;
CREATE TABLE rpa_user_assistant_field
(
    id           BIGSERIAL ,
    assistant_id bigint       NOT NULL ,
    field_key    varchar(128) NOT NULL ,
    field_name   varchar(255)          DEFAULT NULL ,
    field_value  text         NOT NULL ,
    create_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_assistant_field FOREIGN KEY (assistant_id) REFERENCES rpa_user_assistant (id) ON DELETE CASCADE);

CREATE UNIQUE INDEX rpa_user_assistant_field_uk_assistant_field ON rpa_user_assistant_field (assistant_id,field_key);
CREATE INDEX rpa_user_assistant_field_idx_assistant ON rpa_user_assistant_field (assistant_id);

DROP TABLE IF EXISTS share_chat;
CREATE TABLE share_chat
(
    id                 BIGSERIAL ,
    uid                varchar(128)    DEFAULT NULL ,
    url_key            varchar(64)     DEFAULT NULL ,
    chat_id            bigint          DEFAULT NULL ,
    bot_id             bigint          DEFAULT '0' ,
    click_times        int             DEFAULT '0' ,
    max_click_times    int             DEFAULT '-1' ,
    url_status         SMALLINT         DEFAULT '1' ,
    create_time        TIMESTAMP        DEFAULT CURRENT_TIMESTAMP ,
    update_time        TIMESTAMP        DEFAULT CURRENT_TIMESTAMP ,
    enabled_plugin_ids varchar(255)    DEFAULT '' ,
    like_times         int    NOT NULL DEFAULT '0' ,
    ip_location        varchar(32)     DEFAULT '' ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX share_chat_idx_url_key ON share_chat (url_key);
CREATE INDEX share_chat_idx_bot_id ON share_chat (bot_id);
CREATE INDEX share_chat_idx_enabled_plugin_ids ON share_chat (enabled_plugin_ids);
CREATE INDEX share_chat_idx_create_time ON share_chat (create_time);
CREATE INDEX share_chat_idx_uid ON share_chat (uid);

DROP TABLE IF EXISTS share_qa;
CREATE TABLE share_qa
(
    id            BIGSERIAL,
    uid           varchar(128)  DEFAULT NULL ,
    share_chat_id bigint        DEFAULT NULL ,
    message_q     varchar(8000) DEFAULT NULL ,
    message_a     TEXT ,
    sid           varchar(128)  DEFAULT NULL ,
    show_status   SMALLINT       DEFAULT '1' ,
    create_time   TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    req_id        bigint        DEFAULT NULL ,
    req_type      SMALLINT       DEFAULT '0' ,
    req_url       text ,
    resp_id       bigint        DEFAULT '0' ,
    resp_type     varchar(128)  DEFAULT NULL ,
    resp_url      varchar(512)  DEFAULT NULL ,
    chat_key      varchar(64)   DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX share_qa_idx_uid ON share_qa (uid);
CREATE INDEX share_qa_idx_resp_type ON share_qa (resp_type);
CREATE INDEX share_qa_idx_share_chat_id ON share_qa (share_chat_id);
CREATE INDEX "uin_uid_share-chat-id" ON share_qa (uid,share_chat_id);

DROP TABLE IF EXISTS system_user;
CREATE TABLE system_user
(
    id                bigint NOT NULL ,
    nickname          varchar(20)  DEFAULT NULL ,
    login             varchar(20)  DEFAULT NULL ,
    email             varchar(128) DEFAULT NULL ,
    mobile            varchar(20)  DEFAULT NULL ,
    last_login_time   TIMESTAMP     DEFAULT NULL ,
    registration_time TIMESTAMP     DEFAULT NULL ,
    create_time       TIMESTAMP     DEFAULT NULL ,
    update_by         bigint       DEFAULT NULL,
    is_delete         SMALLINT DEFAULT '0' ,
    update_time       TIMESTAMP     DEFAULT NULL,
    source            SMALLINT      DEFAULT '1',
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS tag_info_v2;
CREATE TABLE tag_info_v2
(
    id          BIGSERIAL,
    name        varchar(64)        DEFAULT NULL ,
    type        int                DEFAULT NULL ,
    relation_id varchar(50)        DEFAULT NULL ,
    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uid         varchar(128)       DEFAULT NULL,
    repo_id     bigint             DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX tag_info_v2_type_rel_idx ON tag_info_v2 (type,relation_id);

DROP TABLE IF EXISTS text_node_config;
CREATE TABLE text_node_config
(
    id          BIGSERIAL,
    uid         varchar(128) NOT NULL,
    separator   varchar(255)          DEFAULT NULL,
    comment     varchar(255)          DEFAULT NULL,
    deleted     BOOLEAN       NOT NULL DEFAULT FALSE,
    create_time TIMESTAMP              DEFAULT NULL,
    update_time TIMESTAMP              DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS train_set;
CREATE TABLE train_set
(
    id               BIGSERIAL,
    uid              varchar(128) NOT NULL,
    name             varchar(512) NOT NULL,
    description      varchar(1024)         DEFAULT NULL,
    current_ver      varchar(255)          DEFAULT NULL ,
    ver_count        int                   DEFAULT '0' ,
    deleted          BOOLEAN       NOT NULL DEFAULT FALSE,
    create_time      TIMESTAMP              DEFAULT NULL,
    update_time      TIMESTAMP              DEFAULT NULL,
    application_id   bigint                DEFAULT NULL,
    application_type SMALLINT               DEFAULT NULL,
    node_info        varchar(1024)         DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS train_set_ver;
CREATE TABLE train_set_ver
(
    id           BIGSERIAL,
    train_set_id bigint       NOT NULL,
    ver          varchar(255) NOT NULL ,
    filename     varchar(512)          DEFAULT NULL ,
    storage_addr varchar(512)          DEFAULT NULL ,
    deleted      BOOLEAN       NOT NULL DEFAULT FALSE,
    create_time  TIMESTAMP     NOT NULL,
    update_time  TIMESTAMP     NOT NULL,
    description  varchar(255)          DEFAULT NULL,
    node_info    varchar(1024)         DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS train_set_ver_data;
CREATE TABLE train_set_ver_data
(
    id               BIGSERIAL,
    train_set_ver_id bigint   NOT NULL,
    seq              int           DEFAULT NULL,
    question         varchar(2048) DEFAULT NULL,
    expected_answer  varchar(5096) DEFAULT NULL,
    sid              varchar(256)  DEFAULT NULL,
    create_time      TIMESTAMP NOT NULL,
    deleted          BOOLEAN        DEFAULT FALSE,
    source           SMALLINT       DEFAULT '1' ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS untitled_table;
CREATE TABLE untitled_table
(
    id            SERIAL,
    created_tme   TIMESTAMP NOT NULL                                             DEFAULT CURRENT_TIMESTAMP,
    domain        varchar(255) DEFAULT NULL,
    baseModelId   bigint                                                        DEFAULT NULL,
    baseModelName varchar(255) DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS user_broadcast_read;
CREATE TABLE user_broadcast_read
(
    id              BIGINT NOT NULL  ,
    receiver_uid    varchar(128) NOT NULL ,
    notification_id BIGINT NOT NULL ,
    read_at         TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX user_broadcast_read_idx_receiver_uid ON user_broadcast_read (receiver_uid);
CREATE INDEX user_broadcast_read_idx_notification ON user_broadcast_read (notification_id);

DROP TABLE IF EXISTS user_favorite_tool;
CREATE TABLE user_favorite_tool
(
    id             BIGSERIAL,
    user_id        varchar(128) NOT NULL,
    tool_id        bigint       NOT NULL,
    tool_flag_id   varchar(30)           DEFAULT NULL,
    created_time   timestamp    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted     SMALLINT               DEFAULT '0',
    use_flag       SMALLINT               DEFAULT '0' ,
    mcp_tool_id    varchar(100)          DEFAULT NULL,
    plugin_tool_id varchar(100)          DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX user_favorite_tool_idx_user_favorite_tool_user_id ON user_favorite_tool (user_id);
CREATE INDEX user_favorite_tool_idx_user_favorite_tool_tool_id ON user_favorite_tool (tool_id);

DROP TABLE IF EXISTS user_info;
CREATE TABLE user_info
(
    id             BIGSERIAL ,
    uid            varchar(128)                                                  DEFAULT NULL ,
    username       varchar(255)                                                  DEFAULT NULL ,
    avatar         varchar(512)                                                  DEFAULT NULL ,
    nickname       varchar(255)                                                  DEFAULT NULL ,
    mobile         varchar(255) DEFAULT NULL ,
    account_status SMALLINT                                                       DEFAULT '0' ,
    enterprise_service_type int                                                  DEFAULT '0' ,
    user_agreement SMALLINT                                                       DEFAULT '0' ,
    deleted        SMALLINT                                                       DEFAULT '0' ,
    create_time    TIMESTAMP                                                      DEFAULT CURRENT_TIMESTAMP ,
    update_time    TIMESTAMP                                                      DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX user_info_uid_unique_index ON user_info (uid);
CREATE INDEX user_info_idx_create_time ON user_info (create_time);
CREATE INDEX user_info_index_mobile ON user_info (mobile);
CREATE INDEX user_info_idx_username ON user_info (username);
CREATE INDEX user_info_idx_nickname ON user_info (nickname);
CREATE INDEX user_info_idx_deleted ON user_info (deleted);

DROP TABLE IF EXISTS user_lang_chain_info;
CREATE TABLE user_lang_chain_info
(
    id                  BIGSERIAL ,
    bot_id              int                                                           NOT NULL ,
    name                varchar(255) DEFAULT NULL ,
    desc                text ,
    open                JSONB         DEFAULT NULL ,
    gcy                 JSONB         DEFAULT NULL ,
    uid                 varchar(128) NOT NULL ,
    flow_id             varchar(64)  DEFAULT NULL ,
    space_id            bigint       DEFAULT NULL,
    maas_id             bigint       DEFAULT NULL ,
    bot_name            varchar(255) DEFAULT NULL ,
    extra_inputs        JSONB         DEFAULT NULL ,
    extra_inputs_config JSONB         DEFAULT NULL ,
    create_time         TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time         TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX user_lang_chain_info_idx_bot_id ON user_lang_chain_info (bot_id);
CREATE INDEX user_lang_chain_info_idx_uid ON user_lang_chain_info (uid);

DROP TABLE IF EXISTS user_lang_chain_log;
CREATE TABLE user_lang_chain_log
(
    id          BIGSERIAL,
    bot_id      bigint                                                        DEFAULT NULL,
    maas_id     bigint                                                        DEFAULT NULL,
    flow_id     varchar(64)  DEFAULT NULL,
    uid         varchar(128) DEFAULT NULL,
    space_id    bigint                                                        DEFAULT NULL,
    create_time TIMESTAMP NOT NULL                                             DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP NOT NULL                                             DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id));

CREATE INDEX user_lang_chain_log_idx_uid ON user_lang_chain_log (uid);

DROP TABLE IF EXISTS user_notifications;
CREATE TABLE user_notifications
(
    id              BIGINT NOT NULL  ,
    notification_id BIGINT NOT NULL ,
    receiver_uid    varchar(128) NOT NULL ,
    is_read         SMALLINT                                                       NOT NULL DEFAULT '0' ,
    read_at         TIMESTAMP(3) DEFAULT NULL ,
    received_at     TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    extra           JSONB                                                                   DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX user_notifications_uniq_user_notification ON user_notifications (notification_id,receiver_uid);
CREATE INDEX user_notifications_idx_user_unread ON user_notifications (receiver_uid,is_read,received_at DESC);
CREATE INDEX user_notifications_idx_notification ON user_notifications (notification_id);

DROP TABLE IF EXISTS user_thread_pool_config;
CREATE TABLE user_thread_pool_config
(
    id   BIGSERIAL,
    uid  varchar(100) NOT NULL ,
    size int                                                           NOT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS vcn_info;
CREATE TABLE vcn_info
(
    id          BIGSERIAL,
    vcn         varchar(255)  DEFAULT NULL,
    name        varchar(255)  DEFAULT NULL,
    style       varchar(255)  DEFAULT NULL,
    emt         varchar(255)  DEFAULT NULL,
    image_url   varchar(1024) DEFAULT NULL,
    create_time TIMESTAMP      DEFAULT NULL,
    valid       BOOLEAN        DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS voice_chat_personality_agent;
CREATE TABLE voice_chat_personality_agent
(
    id                      BIGSERIAL,
    uid                     bigint                                                                DEFAULT NULL,
    player_id               varchar(64)          DEFAULT '' ,
    agent_id                varchar(64) NOT NULL DEFAULT '' ,
    vcn_id                  bigint                                                                DEFAULT NULL ,
    agent_name              varchar(64)          DEFAULT '' ,
    agent_type              varchar(16)          DEFAULT '' ,
    player_call             varchar(64)          DEFAULT '' ,
    identity                varchar(100)         DEFAULT '' ,
    personality_description varchar(2000)        DEFAULT '' ,
    image_url               varchar(2250)        DEFAULT '' ,
    is_open                 SMALLINT                                                               DEFAULT NULL ,
    is_del                  SMALLINT                                                               DEFAULT NULL ,
    create_time             TIMESTAMP                                                              DEFAULT CURRENT_TIMESTAMP,
    update_time             TIMESTAMP                                                              DEFAULT CURRENT_TIMESTAMP,
    virtual_url             varchar(2048)        DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX voice_chat_personality_agent_idx_agent_id ON voice_chat_personality_agent (agent_id);
CREATE INDEX voice_chat_personality_agent_idx_agent_name ON voice_chat_personality_agent (agent_name);
CREATE INDEX voice_chat_personality_agent_idx_uid ON voice_chat_personality_agent (uid);
CREATE INDEX voice_chat_personality_agent_idx_vcn_id ON voice_chat_personality_agent (vcn_id);

DROP TABLE IF EXISTS xingchen_official_prompt;
CREATE TABLE xingchen_official_prompt
(
    id             BIGSERIAL ,
    name           varchar(255) NOT NULL ,
    prompt_key     varchar(255) NOT NULL ,
    uid            varchar(128) NOT NULL DEFAULT '0' ,
    type           SMALLINT      NOT NULL DEFAULT '0' ,
    latest_version varchar(50)           DEFAULT '' ,
    model_config   JSONB         NOT NULL ,
    prompt_text    JSONB         NOT NULL ,
    prompt_input   JSONB         NOT NULL ,
    status         SMALLINT      NOT NULL DEFAULT '0' ,
    is_delete      SMALLINT      NOT NULL DEFAULT '0' ,
    commit_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    create_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX xingchen_official_prompt_uk_prompt_key ON xingchen_official_prompt (prompt_key);
CREATE INDEX xingchen_official_prompt_idx_uid ON xingchen_official_prompt (uid);
CREATE INDEX xingchen_official_prompt_idx_type ON xingchen_official_prompt (type);
CREATE INDEX xingchen_official_prompt_idx_status ON xingchen_official_prompt (status);
CREATE INDEX xingchen_official_prompt_idx_create_time ON xingchen_official_prompt (create_time);

DROP TABLE IF EXISTS xingchen_prompt_manage;
CREATE TABLE xingchen_prompt_manage
(
    id              BIGSERIAL ,
    name            varchar(500) NOT NULL ,
    prompt_key      varchar(255) NOT NULL ,
    uid             varchar(128) NOT NULL ,
    type            SMALLINT      NOT NULL DEFAULT '0' ,
    latest_version  varchar(50)           DEFAULT '' ,
    current_version varchar(50)           DEFAULT '' ,
    model_config    JSONB         NOT NULL ,
    prompt_text     JSONB         NOT NULL ,
    prompt_input    JSONB         NOT NULL ,
    status          SMALLINT      NOT NULL DEFAULT '0' ,
    is_delete       SMALLINT      NOT NULL DEFAULT '0' ,
    commit_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    create_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX xingchen_prompt_manage_uk_prompt_key_uid ON xingchen_prompt_manage (prompt_key,uid);
CREATE INDEX xingchen_prompt_manage_idx_uid ON xingchen_prompt_manage (uid);
CREATE INDEX xingchen_prompt_manage_idx_type ON xingchen_prompt_manage (type);
CREATE INDEX xingchen_prompt_manage_idx_status ON xingchen_prompt_manage (status);
CREATE INDEX xingchen_prompt_manage_idx_latest_version ON xingchen_prompt_manage (latest_version);
CREATE INDEX xingchen_prompt_manage_idx_create_time ON xingchen_prompt_manage (create_time);

DROP TABLE IF EXISTS xingchen_prompt_version;
CREATE TABLE xingchen_prompt_version
(
    id           BIGSERIAL ,
    prompt_id    varchar(50)  NOT NULL ,
    uid          varchar(128) NOT NULL ,
    version      varchar(50)  NOT NULL ,
    version_desc text ,
    commit_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    commit_user  varchar(128) NOT NULL ,
    model_config JSONB         NOT NULL ,
    prompt_text  JSONB         NOT NULL ,
    prompt_input JSONB         NOT NULL ,
    is_delete    SMALLINT      NOT NULL DEFAULT '0' ,
    create_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX xingchen_prompt_version_idx_prompt_id ON xingchen_prompt_version (prompt_id);
CREATE INDEX xingchen_prompt_version_idx_uid ON xingchen_prompt_version (uid);
CREATE INDEX xingchen_prompt_version_idx_version ON xingchen_prompt_version (version);
CREATE INDEX xingchen_prompt_version_idx_commit_user ON xingchen_prompt_version (commit_user);
CREATE INDEX xingchen_prompt_version_idx_commit_time ON xingchen_prompt_version (commit_time);
CREATE INDEX xingchen_prompt_version_idx_create_time ON xingchen_prompt_version (create_time);

DROP TABLE IF EXISTS "z-bot_model_config_copy";
CREATE TABLE "z-bot_model_config_copy"
(
    id           BIGSERIAL,
    bot_id       bigint NOT NULL ,
    model_config text   NOT NULL ,
    create_time  timestamp NULL DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS "z-bot_repo_subscript";
CREATE TABLE "z-bot_repo_subscript"
(
    id          BIGSERIAL,
    bot_id      bigint      NOT NULL ,
    app_id      varchar(64) NOT NULL ,
    repo_id     bigint      NOT NULL ,
    create_time timestamp NULL DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS "z-workflow_dialog-bak";
CREATE TABLE "z-workflow_dialog-bak"
(
    id          BIGSERIAL,
    workflow_id bigint   DEFAULT NULL,
    question    text,
    answer      text,
    data        TEXT,
    create_time TIMESTAMP DEFAULT NULL,
    PRIMARY KEY (id)
);

CREATE INDEX z_workflow_dialog_bak_idx_workflow_id ON "z-workflow_dialog-bak" (workflow_id);

DROP TABLE IF EXISTS app_mst;
CREATE TABLE app_mst (
  id           BIGSERIAL,
  uid          varchar(128)   NOT NULL        ,
  app_name     varchar(128)   DEFAULT NULL    ,
  app_describe varchar(512)   DEFAULT NULL    ,
  app_id       varchar(128)   DEFAULT NULL    ,
  app_key      varchar(128)   DEFAULT NULL    ,
  app_secret   varchar(128)   DEFAULT NULL    ,
  is_delete    SMALLINT        DEFAULT '0'     ,
  create_time  TIMESTAMP       DEFAULT NULL    ,
  update_time  TIMESTAMP       DEFAULT NULL    ,
  PRIMARY KEY (id));

CREATE INDEX app_mst_idx_uid ON app_mst (uid);
CREATE INDEX app_mst_idx_app_id ON app_mst (app_id);
CREATE INDEX app_mst_idx_app_name ON app_mst (app_name);

DROP TABLE IF EXISTS personality_category;
CREATE TABLE personality_category
(
    id          BIGSERIAL ,
    name        varchar(64) NOT NULL ,
    sort        int          NOT NULL DEFAULT '0' ,
    deleted     int          NOT NULL DEFAULT '0' ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS personality_role;
CREATE TABLE personality_role
(
    id          BIGSERIAL ,
    name        varchar(255) NOT NULL ,
    description text ,
    head_cover  varchar(2048) NOT NULL ,
    prompt      text ,
    cover       varchar(2048) NOT NULL ,
    sort        int          NOT NULL DEFAULT '0' ,
    category_id bigint       NOT NULL ,
    deleted     int          NOT NULL DEFAULT '0' ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX personality_role_idx_category_id ON personality_role (category_id);

DROP TABLE IF EXISTS personality_config;
CREATE TABLE personality_config
(
    id          BIGSERIAL ,
    bot_id      bigint       NOT NULL ,
    personality text ,
    scene_type  int          DEFAULT NULL ,
    scene_info  varchar(1024) ,
    config_type int          NOT NULL ,
    deleted     int          NOT NULL DEFAULT '0' ,
    enabled     int          NOT NULL DEFAULT '1' ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX personality_config_idx_bot_id ON personality_config (bot_id);
CREATE UNIQUE INDEX personality_config_idx_bot_id_config_type ON personality_config (bot_id, config_type);

DROP TABLE IF EXISTS pronunciation_person_config;
create table pronunciation_person_config
(
    id                 bigint
        primary key,
    name               varchar(64)                        not null ,
    cover_url          varchar(2048)                      null ,
    voice_type         varchar(64)                        null ,
    sort               int      default 0                 null ,
    speaker_type varchar(64)                        null ,
    exquisite          SMALLINT  default 0                 null ,
    deleted            SMALLINT  default 0                 null ,
    create_time        TIMESTAMP default CURRENT_TIMESTAMP null ,
    update_time        TIMESTAMP default CURRENT_TIMESTAMP null
);

DROP TABLE IF EXISTS custom_speaker;
create table custom_speaker
(
    id          bigint
        primary key,
    create_uid  varchar(64)                        not null,
    space_id    bigint                             null,
    name        varchar(64)                        not null,
    task_id     varchar(64)                        not null,
    asset_id    varchar(64)                        null,
    deleted     SMALLINT  default 0                 not null,
    create_time TIMESTAMP default CURRENT_TIMESTAMP null ,
    update_time TIMESTAMP default CURRENT_TIMESTAMP null ,
    constraint uni_task_id
        unique (task_id));

CREATE INDEX custom_speaker_idx_asset_id ON custom_speaker (asset_id);
CREATE INDEX custom_speaker_idx_bot_id ON custom_speaker (space_id);

