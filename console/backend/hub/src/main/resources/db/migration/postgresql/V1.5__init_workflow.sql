-- Migration script for init_workflow

DROP TABLE IF EXISTS flow_db_rel;
CREATE TABLE flow_db_rel
(
    id          BIGSERIAL,
    flow_id     varchar(100) NOT NULL,
    db_id       varchar(100) NOT NULL,
    tb_id       bigint DEFAULT NULL,
    create_time TIMESTAMP     NOT NULL,
    update_time TIMESTAMP     NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS flow_protocol_temp;
CREATE TABLE flow_protocol_temp
(
    flow_id      varchar(255) NOT NULL,
    created_time TIMESTAMP     NOT NULL,
    biz_protocol TEXT   NOT NULL,
    sys_protocol TEXT
);

DROP TABLE IF EXISTS flow_release_aiui_info;
CREATE TABLE flow_release_aiui_info
(
    id   SERIAL,
    data text NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS flow_release_channel;
CREATE TABLE flow_release_channel
(
    flow_id     varchar(255) NOT NULL,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP              DEFAULT CURRENT_TIMESTAMP,
    channel     varchar(255) NOT NULL,
    info_id     bigint                DEFAULT NULL,
    status      SMALLINT               DEFAULT '0' ,
    id          BIGSERIAL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS flow_repo_rel;
CREATE TABLE flow_repo_rel
(
    flow_id     varchar(255) NOT NULL,
    repo_id     varchar(255) NOT NULL,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS flow_tool_rel;
CREATE TABLE flow_tool_rel
(
    flow_id     varchar(255) NOT NULL,
    tool_id     varchar(255) NOT NULL,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version     varchar(100)          DEFAULT NULL
);

DROP TABLE IF EXISTS workflow;
CREATE TABLE workflow
(
    id                   BIGSERIAL ,
    uid                  varchar(128) NOT NULL ,
    app_id               varchar(255) NOT NULL,
    flow_id              varchar(255)          DEFAULT NULL,
    name                 varchar(255) NOT NULL,
    description          varchar(512) NOT NULL,
    deleted              BOOLEAN       NOT NULL DEFAULT FALSE,
    is_public            BOOLEAN       NOT NULL DEFAULT FALSE,
    create_time          TIMESTAMP     NOT NULL,
    update_time          TIMESTAMP              DEFAULT NULL,
    published_data       TEXT,
    data                 TEXT,
    avatar_icon          varchar(1000)         DEFAULT NULL,
    avatar_color         varchar(255)          DEFAULT NULL,
    status               SMALLINT      NOT NULL DEFAULT '-1' ,
    can_publish          BOOLEAN                DEFAULT FALSE,
    app_updatable        BOOLEAN                DEFAULT FALSE,
    top                  BOOLEAN                DEFAULT FALSE,
    edge_type            varchar(255)          DEFAULT NULL,
    order                int                   DEFAULT '0',
    eval_set_id          bigint                DEFAULT NULL,
    source               SMALLINT               DEFAULT '1',
    bak                  TEXT,
    editing              BOOLEAN                DEFAULT TRUE,
    eval_page_first_time text,
    advanced_config      text ,
    ext                  text,
    category             int                   DEFAULT NULL ,
    space_id             bigint                DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX workflow_flow_id ON workflow (flow_id);

DROP TABLE IF EXISTS workflow_comparison;
CREATE TABLE workflow_comparison
(
    id          BIGSERIAL,
    flow_id     varchar(100) NOT NULL ,
    type        SMALLINT      NOT NULL DEFAULT '0' ,
    data        TEXT   NOT NULL ,
    create_time TIMESTAMP     NOT NULL ,
    update_time TIMESTAMP     NOT NULL ,
    prompt_id   varchar(100) NOT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS workflow_dialog;
CREATE TABLE workflow_dialog
(
    id            BIGSERIAL,
    uid           varchar(128)     DEFAULT NULL,
    workflow_id   bigint           DEFAULT NULL,
    question      text,
    answer        TEXT,
    data          TEXT,
    create_time   TIMESTAMP         DEFAULT NULL,
    deleted       BOOLEAN           DEFAULT FALSE,
    sid           varchar(255)     DEFAULT NULL,
    type          SMALLINT NOT NULL DEFAULT '1' ,
    question_item text,
    answer_item   TEXT,
    chat_id       varchar(100)     DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX workflow_dialog_workflow_id ON workflow_dialog (workflow_id);

DROP TABLE IF EXISTS workflow_dialog_bak;
CREATE TABLE workflow_dialog_bak
(
    id          BIGSERIAL,
    uid         varchar(128) DEFAULT NULL,
    workflow_id bigint       DEFAULT NULL,
    question    text,
    answer      text,
    data        TEXT,
    create_time TIMESTAMP     DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX workflow_dialog_bak_workflow_id ON workflow_dialog_bak (workflow_id);

DROP TABLE IF EXISTS workflow_feedback;
CREATE TABLE workflow_feedback
(
    id          BIGSERIAL,
    uid         varchar(128) NOT NULL ,
    user_name   varchar(100) NOT NULL ,
    bot_id      varchar(100) NOT NULL,
    flow_id     varchar(100) NOT NULL,
    sid         varchar(100) NOT NULL,
    start_time  TIMESTAMP      DEFAULT NULL,
    end_time    TIMESTAMP      DEFAULT NULL,
    cost_time   int           DEFAULT NULL ,
    token       int           DEFAULT NULL ,
    status      varchar(100)  DEFAULT NULL ,
    error_code  varchar(100)  DEFAULT NULL,
    pic_url     text ,
    description varchar(1024) DEFAULT NULL ,
    create_time TIMESTAMP      DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS workflow_node_history;
CREATE TABLE workflow_node_history
(
    id           BIGSERIAL,
    node_id      varchar(255) NOT NULL,
    chat_id      varchar(255) DEFAULT NULL,
    raw_question text,
    raw_answer   text,
    create_time  TIMESTAMP     NOT NULL,
    flow_id      varchar(255) DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX workflow_node_history_node_id ON workflow_node_history (node_id);
CREATE INDEX workflow_node_history_chat_id ON workflow_node_history (chat_id);

DROP TABLE IF EXISTS workflow_template_group;
CREATE TABLE workflow_template_group
(
    id            SERIAL ,
    create_user   varchar(32) NOT NULL ,
    group_name    varchar(20) NOT NULL ,
    sort_index    int         NOT NULL ,
    is_delete     SMALLINT     NOT NULL DEFAULT '0' ,
    create_time   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    group_name_en varchar(128)         DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX workflow_template_group_idx_group_name ON workflow_template_group (group_name);

DROP TABLE IF EXISTS workflow_version;
CREATE TABLE workflow_version
(
    id               BIGSERIAL,
    name             varchar(100)          DEFAULT NULL ,
    version_num      varchar(100) NOT NULL ,
    data             TEXT ,
    flow_id          varchar(19)  NOT NULL,
    is_deleted       int          NOT NULL DEFAULT '0' ,
    deleted          int          NOT NULL DEFAULT '1' ,
    created_time     TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    updated_time     TIMESTAMP              DEFAULT CURRENT_TIMESTAMP,
    is_current       int          NOT NULL DEFAULT '1' ,
    is_version       int          NOT NULL DEFAULT '1' ,
    sys_data         TEXT ,
    description      varchar(100)          DEFAULT NULL ,
    publish_channels varchar(255)          DEFAULT NULL ,
    publish_channel  int                   DEFAULT NULL ,
    publish_result   text ,
    bot_id           varchar(100)          DEFAULT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE workflow_config (
                                   id bigint(20) NOT NULL ,
                                   name varchar(100) DEFAULT NULL ,
                                   version_num varchar(100) NOT NULL DEFAULT '-1' ,
                                   flow_id varchar(19) NOT NULL ,
                                   bot_id int(11) DEFAULT NULL,
                                   config TEXT ,
                                   created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ,
                                   updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                   deleted SMALLINT DEFAULT '0' ,
                                   PRIMARY KEY (id)
);

