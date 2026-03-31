-- Migration script for init_bot

DROP TABLE IF EXISTS bot_chat_file_param;
CREATE TABLE bot_chat_file_param
(
    id          BIGSERIAL ,
    uid         varchar(128) NOT NULL ,
    chat_id     bigint                                                        NOT NULL ,
    name        varchar(255) NOT NULL ,
    file_ids    JSONB     DEFAULT NULL ,
    file_urls   JSONB     DEFAULT NULL ,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ,
    is_delete   SMALLINT  DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX bot_chat_file_param_idx_uid ON bot_chat_file_param (uid);
CREATE INDEX bot_chat_file_param_idx_chat_id ON bot_chat_file_param (chat_id);
CREATE INDEX bot_chat_file_param_idx_name ON bot_chat_file_param (name);

DROP TABLE IF EXISTS bot_conversation_stats;
CREATE TABLE bot_conversation_stats
(
    id                BIGSERIAL ,
    uid               varchar(128) NOT NULL ,
    space_id          bigint                                                                 DEFAULT NULL ,
    bot_id            int                                                           NOT NULL ,
    chat_id           bigint                                                        NOT NULL ,
    sid               varchar(255)          DEFAULT NULL ,
    token_consumed    int                                                           NOT NULL DEFAULT '0' ,
    conversation_date date                                                          NOT NULL ,
    create_time       TIMESTAMP                                                      NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    is_delete         SMALLINT                                                       NOT NULL DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX bot_conversation_stats_idx_bot_id_date ON bot_conversation_stats (bot_id,conversation_date);
CREATE INDEX bot_conversation_stats_idx_uid_bot_id ON bot_conversation_stats (uid,bot_id);
CREATE INDEX bot_conversation_stats_idx_space_id_bot_id ON bot_conversation_stats (space_id,bot_id);
CREATE INDEX bot_conversation_stats_idx_chat_id ON bot_conversation_stats (chat_id);
CREATE INDEX bot_conversation_stats_idx_create_time ON bot_conversation_stats (create_time);

DROP TABLE IF EXISTS bot_dataset;
CREATE TABLE bot_dataset
(
    id            BIGSERIAL,
    bot_id        bigint NOT NULL ,
    dataset_id    bigint       DEFAULT NULL ,
    dataset_index varchar(255) DEFAULT NULL ,
    is_act        SMALLINT      DEFAULT '1' ,
    create_time   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    uid           varchar(128) DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX bot_dataset_idx_id_bot_id ON bot_dataset (id,bot_id);
CREATE INDEX bot_dataset_idx_uid ON bot_dataset (uid);
CREATE INDEX bot_dataset_idx_is_act ON bot_dataset (is_act);
CREATE INDEX bot_dataset_idx_create_time ON bot_dataset (create_time);

DROP TABLE IF EXISTS bot_dataset_maas;
CREATE TABLE bot_dataset_maas
(
    id            BIGSERIAL,
    bot_id        bigint NOT NULL ,
    dataset_id    bigint       DEFAULT NULL ,
    dataset_index varchar(255) DEFAULT NULL ,
    is_act        SMALLINT      DEFAULT '1' ,
    create_time   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    uid           varchar(128) DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX bot_dataset_maas_idx_id_bot_id ON bot_dataset_maas (id,bot_id);
CREATE INDEX bot_dataset_maas_idx_uid ON bot_dataset_maas (uid);
CREATE INDEX bot_dataset_maas_idx_is_act ON bot_dataset_maas (is_act);
CREATE INDEX bot_dataset_maas_idx_create_time ON bot_dataset_maas (create_time);

DROP TABLE IF EXISTS bot_favorite;
CREATE TABLE bot_favorite
(
    id          BIGSERIAL,
    uid         varchar(128) NOT NULL,
    bot_id      int          NOT NULL,
    create_time TIMESTAMP DEFAULT NULL,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id));

CREATE INDEX bot_favorite_idx_uid ON bot_favorite (uid);

DROP TABLE IF EXISTS bot_flow_rel;
CREATE TABLE bot_flow_rel
(
    id          SERIAL,
    create_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    flow_id     varchar(255) DEFAULT NULL,
    bot_id      bigint       DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS bot_model_bind;
CREATE TABLE bot_model_bind
(
    id             BIGSERIAL,
    uid            varchar(128) NOT NULL,
    bot_id         bigint                DEFAULT NULL,
    app_id         varchar(255) NOT NULL,
    llm_service_id varchar(255) NOT NULL,
    domain         varchar(255) NOT NULL,
    patch_id       varchar(255) NOT NULL DEFAULT '0',
    model_name     varchar(255)          DEFAULT NULL,
    create_time    TIMESTAMP              DEFAULT NULL,
    model_type     SMALLINT               DEFAULT '1',
    PRIMARY KEY (id),
    UNIQUE KEY bot_id (bot_id,app_id(191),llm_service_id(191),domain(191),patch_id(191))
);

DROP TABLE IF EXISTS bot_model_config;
CREATE TABLE bot_model_config
(
    id           BIGSERIAL,
    bot_id       bigint NOT NULL ,
    model_config text   NOT NULL ,
    create_time  TIMESTAMP DEFAULT NULL,
    update_time  TIMESTAMP DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS bot_offiaccount;
CREATE TABLE bot_offiaccount
(
    id           BIGSERIAL,
    uid          varchar(128) DEFAULT NULL ,
    bot_id       bigint       DEFAULT NULL ,
    appid        varchar(100) DEFAULT NULL ,
    release_type SMALLINT      DEFAULT '1' ,
    status       SMALLINT      DEFAULT '0' ,
    create_time  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX bot_offiaccount_bot_id_index ON bot_offiaccount (bot_id);
CREATE INDEX bot_offiaccount_uid_index ON bot_offiaccount (uid);

DROP TABLE IF EXISTS bot_offiaccount_chat;
CREATE TABLE bot_offiaccount_chat
(
    id          BIGSERIAL,
    app_id      varchar(64) DEFAULT NULL ,
    open_id     varchar(64) DEFAULT NULL ,
    msg_id      bigint      DEFAULT NULL ,
    req         text ,
    resp        text ,
    sid         varchar(64) DEFAULT NULL ,
    create_time TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX bot_offiaccount_chat_index_app_id ON bot_offiaccount_chat (app_id);
CREATE INDEX bot_offiaccount_chat_index_open_id ON bot_offiaccount_chat (open_id);
CREATE INDEX bot_offiaccount_chat_index_msg_id ON bot_offiaccount_chat (msg_id);

DROP TABLE IF EXISTS bot_offiaccount_record;
CREATE TABLE bot_offiaccount_record
(
    id          BIGSERIAL,
    bot_id      bigint       DEFAULT NULL ,
    appid       varchar(100) DEFAULT NULL ,
    auth_type   SMALLINT      DEFAULT NULL ,
    create_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX bot_offiaccount_record_appid_index ON bot_offiaccount_record (appid);
CREATE INDEX bot_offiaccount_record_bot_id_index ON bot_offiaccount_record (bot_id);

DROP TABLE IF EXISTS bot_repo_rel;
CREATE TABLE bot_repo_rel
(
    id          BIGSERIAL,
    bot_id      bigint       NOT NULL ,
    app_id      varchar(64)  NOT NULL ,
    repo_id     varchar(200) NOT NULL ,
    file_ids    varchar(500) DEFAULT NULL ,
    create_time timestamp NULL DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS bot_tool_rel;
CREATE TABLE bot_tool_rel
(
    id          BIGSERIAL,
    bot_id      bigint       NOT NULL ,
    tool_id     varchar(100) NOT NULL ,
    create_time timestamp NULL DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS bot_type_list;
CREATE TABLE bot_type_list
(
    id           SERIAL ,
    type_key     int          DEFAULT NULL ,
    type_name    varchar(255) DEFAULT NULL ,
    order_num    int          DEFAULT '0' ,
    show_index   SMALLINT      DEFAULT '0' ,
    is_act       SMALLINT      DEFAULT '1' ,
    create_time  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    icon         varchar(500) DEFAULT '' ,
    type_name_en varchar(128) DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS chat_bot_api;
CREATE TABLE chat_bot_api
(
    id           BIGSERIAL,
    uid          varchar(128)  NOT NULL ,
    bot_id       int           NOT NULL ,
    assistant_id varchar(32)   NOT NULL ,
    app_id       varchar(32)  DEFAULT NULL ,
    api_secret   varchar(64)   NOT NULL ,
    api_key      varchar(64)   NOT NULL ,
    api_path     varchar(32)   NOT NULL ,
    prompt       varchar(2048) NOT NULL ,
    plugin_id    varchar(256)  NOT NULL ,
    embedding_id varchar(256)  NOT NULL ,
    description  varchar(256) DEFAULT NULL ,
    create_time  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX chat_bot_api_idx_assistant_id ON chat_bot_api (assistant_id);
CREATE INDEX chat_bot_api_idx_bot_id ON chat_bot_api (bot_id);
CREATE INDEX chat_bot_api_idx_uid ON chat_bot_api (uid);
CREATE INDEX chat_bot_api_idx_create_time ON chat_bot_api (create_time);

DROP TABLE IF EXISTS chat_bot_base;
CREATE TABLE chat_bot_base
(
    id                SERIAL ,
    uid               varchar(128)     DEFAULT NULL ,
    bot_name          varchar(48)      DEFAULT NULL ,
    bot_type          SMALLINT          DEFAULT NULL ,
    avatar            varchar(1024)    DEFAULT NULL ,
    pc_background     varchar(512)     DEFAULT '' ,
    app_background    varchar(512)     DEFAULT '' ,
    background_color  SMALLINT          DEFAULT '0' ,
    prompt            varchar(2048)    DEFAULT NULL ,
    prologue          varchar(512)     DEFAULT NULL ,
    bot_desc          varchar(255)     DEFAULT NULL ,
    is_delete         SMALLINT          DEFAULT '0' ,
    create_time       TIMESTAMP         DEFAULT CURRENT_TIMESTAMP ,
    update_time       TIMESTAMP         DEFAULT CURRENT_TIMESTAMP ,
    support_context   SMALLINT NOT NULL DEFAULT '0' ,
    bot_template      varchar(255)     DEFAULT '' ,
    prompt_type       SMALLINT unsigned NOT NULL DEFAULT '0' ,
    input_example     varchar(600)     DEFAULT '' ,
    botweb_status     SMALLINT NOT NULL DEFAULT '0' ,
    version           int              DEFAULT '1' ,
    support_document  SMALLINT          DEFAULT '0' ,
    support_system    SMALLINT          DEFAULT '0' ,
    prompt_system     SMALLINT          DEFAULT '0' ,
    support_upload    SMALLINT NOT NULL DEFAULT '0' ,
    bot_name_en       varchar(48)      DEFAULT NULL ,
    bot_desc_en       varchar(500)     DEFAULT NULL ,
    client_type       SMALLINT NOT NULL DEFAULT '0' ,
    vcn_cn            varchar(32)      DEFAULT NULL ,
    vcn_en            varchar(32)      DEFAULT NULL ,
    vcn_speed         SMALLINT NOT NULL DEFAULT '50' ,
    is_sentence       SMALLINT NOT NULL DEFAULT '0' ,
    opened_tool       varchar(128)     DEFAULT 'ifly_search,text_to_image,codeinterpreter' ,
    client_hide       varchar(10)      DEFAULT '' ,
    virtual_bot_type  SMALLINT          DEFAULT NULL ,
    virtual_agent_id  bigint           DEFAULT NULL ,
    style             int              DEFAULT NULL ,
    background        varchar(512)     DEFAULT NULL ,
    virtual_character varchar(512)     DEFAULT NULL ,
    model             varchar(32)      DEFAULT 'spark' ,
    maas_bot_id       varchar(50)      DEFAULT NULL ,
    prologue_en       varchar(1024)    DEFAULT NULL ,
    input_example_en  varchar(1024)    DEFAULT NULL ,
    space_id          bigint           DEFAULT NULL ,
    model_id          bigint           DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX chat_bot_base_idx_create_time ON chat_bot_base (create_time);
CREATE INDEX chat_bot_base_idx_support_context ON chat_bot_base (support_context);
CREATE INDEX chat_bot_base_idx_uid ON chat_bot_base (uid);
CREATE INDEX chat_bot_base_idx_botweb_status ON chat_bot_base (botweb_status);
CREATE INDEX chat_bot_base_idx_space_id ON chat_bot_base (space_id);

DROP TABLE IF EXISTS chat_bot_list;
CREATE TABLE chat_bot_list
(
    id              SERIAL,
    uid             varchar(128)     DEFAULT NULL ,
    market_bot_id   int              DEFAULT '0' ,
    real_bot_id     int              DEFAULT '0' ,
    name            varchar(48)      DEFAULT NULL ,
    bot_type        SMALLINT          DEFAULT '1' ,
    avatar          varchar(1024)    DEFAULT NULL ,
    prompt          varchar(2048)    DEFAULT NULL ,
    bot_desc        varchar(255)     DEFAULT NULL ,
    is_act          SMALLINT          DEFAULT '1' ,
    create_time     TIMESTAMP         DEFAULT CURRENT_TIMESTAMP ,
    update_time     TIMESTAMP         DEFAULT CURRENT_TIMESTAMP ,
    support_context SMALLINT NOT NULL DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX chat_bot_list_idx_act ON chat_bot_list (is_act);
CREATE INDEX chat_bot_list_idx_create_time2 ON chat_bot_list (create_time);
CREATE INDEX chat_bot_list_idx_real_bot_id ON chat_bot_list (real_bot_id);
CREATE INDEX chat_bot_list_idx_uid ON chat_bot_list (uid);

DROP TABLE IF EXISTS chat_bot_market;
CREATE TABLE chat_bot_market
(
    id               SERIAL,
    bot_id           int                                                          DEFAULT NULL ,
    uid              varchar(128)                                                 DEFAULT NULL ,
    bot_name         varchar(48)                                                  DEFAULT NULL ,
    bot_type         SMALLINT                                                      DEFAULT '1' ,
    avatar           varchar(1024)                                                DEFAULT NULL ,
    pc_background    varchar(512)                                                 DEFAULT '' ,
    app_background   varchar(512)                                                 DEFAULT '' ,
    background_color SMALLINT                                                      DEFAULT '0' ,
    prompt           varchar(2048)                                                DEFAULT NULL ,
    prologue         varchar(512)                                                 DEFAULT NULL ,
    show_others      SMALLINT                                                      DEFAULT NULL ,
    bot_desc         varchar(255)                                                 DEFAULT NULL ,
    bot_status       SMALLINT                                                      DEFAULT '1' ,
    block_reason     varchar(255)                                                 DEFAULT NULL ,
    hot_num          int                                                          DEFAULT '0' ,
    is_delete        SMALLINT                                                      DEFAULT '0' ,
    show_index       SMALLINT                                                      DEFAULT '0' ,
    sort_hot         int                                                          DEFAULT '0' ,
    sort_latest      int                                                          DEFAULT '0' ,
    audit_time       TIMESTAMP                                                     DEFAULT NULL ,
    create_time      TIMESTAMP                                                     DEFAULT CURRENT_TIMESTAMP ,
    update_time      TIMESTAMP                                                     DEFAULT CURRENT_TIMESTAMP ,
    support_context  SMALLINT NOT NULL                                             DEFAULT '0' ,
    version          int                                                          DEFAULT '1' ,
    show_weight      int                                                          DEFAULT '1' ,
    score            int                                                          DEFAULT NULL ,
    client_hide      varchar(10)                                                  DEFAULT '' ,
    model            varchar(64) DEFAULT NULL ,
    opened_tool      varchar(255)                                                 DEFAULT NULL ,
    publish_channels varchar(255)                                                 DEFAULT NULL ,
    model_id         bigint                                                       DEFAULT NULL ,
    support_document  SMALLINT NOT NULL                                             DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX chat_bot_market_idx_bot_id ON chat_bot_market (bot_id);
CREATE INDEX chat_bot_market_idx_create_time3 ON chat_bot_market (create_time);
CREATE INDEX chat_bot_market_uid_index ON chat_bot_market (uid);
CREATE INDEX chat_bot_market_idx_bot_status ON chat_bot_market (bot_status,bot_id);

DROP TABLE IF EXISTS chat_bot_prompt_struct;
CREATE TABLE chat_bot_prompt_struct
(
    id           BIGSERIAL,
    bot_id       int                                                            NOT NULL ,
    prompt_key   varchar(64)   NOT NULL ,
    prompt_value varchar(2550) NOT NULL DEFAULT '' ,
    create_time  TIMESTAMP                                                                DEFAULT NULL,
    update_time  TIMESTAMP                                                                DEFAULT NULL,
    PRIMARY KEY (id));

CREATE INDEX chat_bot_prompt_struct_idx_bot_id ON chat_bot_prompt_struct (bot_id);

DROP TABLE IF EXISTS chat_bot_remove;
CREATE TABLE chat_bot_remove
(
    id           SERIAL,
    bot_id       int           DEFAULT NULL ,
    uid          varchar(128)  DEFAULT NULL ,
    bot_name     varchar(48)   DEFAULT NULL ,
    bot_type     SMALLINT       DEFAULT '1' ,
    avatar       varchar(512)  DEFAULT NULL ,
    prompt       varchar(2048) DEFAULT NULL ,
    bot_desc     varchar(255)  DEFAULT NULL ,
    block_reason varchar(255)  DEFAULT NULL ,
    is_delete    SMALLINT       DEFAULT '0' ,
    audit_time   TIMESTAMP      DEFAULT NULL ,
    create_time  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    update_time  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX chat_bot_remove_idx_bot_id ON chat_bot_remove (bot_id);
CREATE INDEX chat_bot_remove_idx_bot_type ON chat_bot_remove (bot_type);
CREATE INDEX chat_bot_remove_idx_create_time4 ON chat_bot_remove (create_time);
CREATE INDEX chat_bot_remove_uid_index ON chat_bot_remove (uid);

DROP TABLE IF EXISTS create_bot_context;
CREATE TABLE create_bot_context
(
    chat_id      varchar(255) NOT NULL,
    step         SMALLINT  DEFAULT NULL,
    biz_data     JSONB     DEFAULT NULL,
    create_time  TIMESTAMP DEFAULT NULL,
    update_time  TIMESTAMP DEFAULT NULL,
    chat_history text,
    PRIMARY KEY (chat_id)
);

DROP TABLE IF EXISTS spark_bot;
CREATE TABLE spark_bot
(
    id             BIGSERIAL ,
    uuid           varchar(64)          DEFAULT NULL,
    name           varchar(64) NOT NULL ,
    user_id        varchar(20)          DEFAULT NULL,
    app_id         varchar(50) NOT NULL,
    description    varchar(255)         DEFAULT NULL ,
    avatar_icon    varchar(255)         DEFAULT NULL ,
    color          varchar(10)          DEFAULT NULL,
    floating_icon  varchar(255)         DEFAULT NULL ,
    greeting       varchar(128)         DEFAULT NULL ,
    floated        SMALLINT DEFAULT '0' ,
    deleted        SMALLINT NOT NULL DEFAULT '0' ,
    create_time    timestamp NULL DEFAULT NULL ,
    update_time    timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    recommend_ques text,
    is_public      SMALLINT     NOT NULL DEFAULT '0' ,
    bot_tag        varchar(100)         DEFAULT NULL ,
    user_count     int                  DEFAULT '0' ,
    dialog_count   int                  DEFAULT '0' ,
    favorite_count int                  DEFAULT '0' ,
    public_id      bigint               DEFAULT NULL ,
    app_updatable  BOOLEAN               DEFAULT FALSE,
    top            BOOLEAN               DEFAULT FALSE,
    eval_set_id    bigint               DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS user_favorite_bot;
CREATE TABLE user_favorite_bot
(
    id           BIGSERIAL,
    user_id      bigint    NOT NULL,
    bot_id       bigint    NOT NULL,
    created_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    use_flag     SMALLINT            DEFAULT '0',
    is_deleted   SMALLINT            DEFAULT '0',
    PRIMARY KEY (id),
    CONSTRAINT user_favorite_bot_ibfk_1 FOREIGN KEY (user_id) REFERENCES system_user (id),
    CONSTRAINT user_favorite_bot_ibfk_2 FOREIGN KEY (bot_id) REFERENCES spark_bot (id));

CREATE INDEX user_favorite_bot_idx_user_favorite_bot_user_id ON user_favorite_bot (user_id);
CREATE INDEX user_favorite_bot_idx_user_favorite_bot_bot_id ON user_favorite_bot (bot_id);

CREATE TABLE IF NOT EXISTS bot_template (
    id INT PRIMARY KEY  ,
    bot_name VARCHAR(32) NOT NULL ,
    bot_desc VARCHAR(200) ,
    bot_template TEXT ,
    bot_type INT NOT NULL ,
    bot_type_name VARCHAR(50) ,
    input_example TEXT ,
    prompt TEXT ,
    prompt_struct_list TEXT ,
    prompt_type INT DEFAULT 0 ,
    support_context INT DEFAULT 0 ,
    bot_status INT DEFAULT 1 ,
    language VARCHAR(10) DEFAULT 'zh' ,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_bot_status (bot_status),
    INDEX idx_bot_type (bot_type),
    INDEX idx_language (language),
    INDEX idx_status_lang (bot_status, language)
);

