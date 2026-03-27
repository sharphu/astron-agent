-- Migration script for init_model

DROP TABLE IF EXISTS base_model_map;
CREATE TABLE base_model_map
(
    id              INT IDENTITY(1,1),
    create_time     TIMESTAMP NOT NULL                                             DEFAULT CURRENT_TIMESTAMP,
    domain          varchar(255) DEFAULT NULL,
    base_model_id   bigint                                                        DEFAULT NULL,
    base_model_name varchar(255) DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS chat_req_model;
CREATE TABLE chat_req_model
(
    id          INT IDENTITY(1,1),
    uid         varchar(128) NOT NULL ,
    chat_id     bigint                DEFAULT NULL ,
    chat_req_id bigint       NOT NULL ,
    type        SMALLINT      NOT NULL DEFAULT '1' ,
    url         varchar(2048)         DEFAULT NULL ,
    status      SMALLINT      NOT NULL DEFAULT '0' ,
    need_his    SMALLINT               DEFAULT '1' ,
    img_desc    varchar(2048)         DEFAULT NULL ,
    intention   varchar(255)          DEFAULT NULL ,
    ocr_result  text ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    data_id     varchar(64)           DEFAULT NULL ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_req_model_idx_uid ON chat_req_model (uid);
CREATE INDEX chat_req_model_idx_req_id ON chat_req_model (chat_req_id);

DROP TABLE IF EXISTS chat_resp_model;
CREATE TABLE chat_resp_model
(
    id          BIGINT IDENTITY(1,1),
    uid         varchar(128) NOT NULL ,
    chat_id     bigint                DEFAULT NULL ,
    req_id      bigint       NOT NULL ,
    content     varchar(8000)         DEFAULT NULL ,
    type        varchar(32)  NOT NULL DEFAULT 'text' ,
    need_his    SMALLINT               DEFAULT '1' ,
    url         text ,
    status      SMALLINT      NOT NULL DEFAULT '0' ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    data_id     varchar(64)           DEFAULT NULL ,
    water_url   text ,
    PRIMARY KEY (id, create_time));

CREATE INDEX chat_resp_model_idx_uid ON chat_resp_model (uid);
CREATE INDEX chat_resp_model_idx_chat_id ON chat_resp_model (chat_id);
CREATE INDEX chat_resp_model_idx_create_time ON chat_resp_model (create_time);
CREATE INDEX chat_resp_model_idx_req_id ON chat_resp_model (req_id);

DROP TABLE IF EXISTS model;
CREATE TABLE model
(
    id                BIGINT IDENTITY(1,1) ,
    name              varchar(255)          DEFAULT NULL ,
    desc              varchar(1024)         DEFAULT NULL ,
    source            int                   DEFAULT NULL ,
    uid               varchar(128) NOT NULL ,
    type              int                   DEFAULT NULL ,
    url               varchar(255)          DEFAULT NULL ,
    domain            varchar(100)          DEFAULT NULL ,
    api_key           varchar(255)          DEFAULT NULL,
    sub_type          bigint                DEFAULT NULL ,
    content           text ,
    is_deleted        BOOLEAN       NOT NULL DEFAULT FALSE ,
    image_url         varchar(255)          DEFAULT NULL,
    doc_url           varchar(255)          DEFAULT NULL,
    remark            varchar(255)          DEFAULT NULL,
    sort              int                   DEFAULT '0' ,
    channel           varchar(255)          DEFAULT '0' ,
    tag               varchar(255)          DEFAULT NULL ,
    color             varchar(100)          DEFAULT NULL ,
    create_time       TIMESTAMP              DEFAULT NULL,
    update_time       TIMESTAMP              DEFAULT NULL,
    config            text ,
    space_id          bigint                DEFAULT NULL ,
    enable            BOOLEAN                DEFAULT TRUE ,
    status            int                   DEFAULT NULL,
    accelerator_count int                   DEFAULT NULL ,
    replica_count     int                   DEFAULT NULL ,
    model_path        varchar(100)          DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS model_category;
CREATE TABLE model_category
(
    id          BIGINT IDENTITY(1,1),
    pid         bigint       NOT NULL,
    key         varchar(100) NOT NULL DEFAULT '',
    name        varchar(255) NOT NULL,
    is_delete   SMALLINT unsigned NOT NULL DEFAULT '0',
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sort_order  int          NOT NULL DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX model_category_idx_key_pid_delete ON model_category (key,pid,is_delete);

DROP TABLE IF EXISTS model_category_rel;
CREATE TABLE model_category_rel
(
    id          BIGINT IDENTITY(1,1),
    model_id    bigint   NOT NULL,
    category_id bigint   NOT NULL,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX model_category_rel_uk_model_id_category_id ON model_category_rel (model_id,category_id);
CREATE INDEX model_category_rel_idx_category ON model_category_rel (category_id);
CREATE INDEX model_category_rel_idx_model ON model_category_rel (model_id);

DROP TABLE IF EXISTS model_common;
CREATE TABLE model_common
(
    id             BIGINT IDENTITY(1,1),
    name           varchar(128) NOT NULL DEFAULT '',
    desc           varchar(500)          DEFAULT NULL ,
    intro          varchar(255) NOT NULL DEFAULT '' ,
    user_name      varchar(64)  NOT NULL DEFAULT '' ,
    user_avatar    varchar(255) NOT NULL DEFAULT '' ,
    service_id     varchar(128) NOT NULL DEFAULT '',
    server_id      varchar(128) NOT NULL DEFAULT '',
    domain         varchar(128) NOT NULL DEFAULT '',
    lic_channel    varchar(128) NOT NULL DEFAULT '',
    llm_source     varchar(128) NOT NULL DEFAULT '',
    url            varchar(128) NOT NULL DEFAULT '',
    model_type     SMALLINT      NOT NULL DEFAULT '0',
    type           SMALLINT      NOT NULL DEFAULT '0',
    source         SMALLINT      NOT NULL DEFAULT '0',
    is_think       SMALLINT      NOT NULL DEFAULT '0',
    multi_mode     SMALLINT      NOT NULL DEFAULT '0',
    is_delete      SMALLINT      NOT NULL DEFAULT '0',
    create_by      bigint       NOT NULL DEFAULT '0',
    create_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_by      bigint       NOT NULL DEFAULT '0',
    update_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uid            varchar(128)          DEFAULT NULL ,
    disclaimer     varchar(2048)         DEFAULT '' ,
    config         text ,
    shelf_status   int                   DEFAULT '0' ,
    shelf_off_time TIMESTAMP              DEFAULT NULL ,
    http_url       varchar(100)          DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS model_custom_category;
CREATE TABLE model_custom_category
(
    id           BIGINT IDENTITY(1,1),
    owner_uid    varchar(128) NOT NULL ,
    key          varchar(100) NOT NULL DEFAULT '' ,
    name         varchar(255) NOT NULL,
    pid          bigint                DEFAULT NULL ,
    normalized   varchar(255) GENERATED ALWAYS AS (lower(trim(name))) VIRTUAL,
    audit_status SMALLINT unsigned NOT NULL DEFAULT '1' ,
    is_delete    SMALLINT unsigned NOT NULL DEFAULT '0',
    create_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id));

CREATE INDEX model_custom_category_idx_key_status ON model_custom_category (key,audit_status);
CREATE INDEX model_custom_category_idx_owner ON model_custom_category (owner_uid);

DROP TABLE IF EXISTS model_custom_category_rel;
CREATE TABLE model_custom_category_rel
(
    id          BIGINT IDENTITY(1,1),
    model_id    bigint   NOT NULL,
    custom_id   bigint   NOT NULL,
    create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_rel_custom FOREIGN KEY (custom_id) REFERENCES model_custom_category (id) ON DELETE CASCADE ON UPDATE CASCADE);

CREATE UNIQUE INDEX model_custom_category_rel_uk_model_custom ON model_custom_category_rel (model_id,custom_id);
CREATE INDEX model_custom_category_rel_idx_custom ON model_custom_category_rel (custom_id);

DROP TABLE IF EXISTS model_list_config;
CREATE TABLE model_list_config
(
    id            INT IDENTITY(1,1),
    create_time   timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    node_type     varchar(255)       DEFAULT NULL,
    name          varchar(255)       DEFAULT NULL,
    description   varchar(255)       DEFAULT NULL,
    tag           varchar(255)       DEFAULT NULL,
    deleted       BOOLEAN             DEFAULT FALSE,
    base_model_id bigint             DEFAULT NULL,
    recommended   BOOLEAN             DEFAULT FALSE,
    domain        varchar(255)       DEFAULT NULL,
    PRIMARY KEY (id)
);

