-- Migration script for init_knowledge

DROP TABLE IF EXISTS chat_file_req;
CREATE TABLE chat_file_req
(
    id            BIGSERIAL,
    file_id       varchar(64)  NOT NULL ,
    chat_id       bigint       NOT NULL ,
    req_id        bigint                DEFAULT NULL ,
    uid           varchar(128) NOT NULL ,
    create_time   TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    client_type   SMALLINT      NOT NULL DEFAULT '0' ,
    deleted       SMALLINT      NOT NULL DEFAULT '0' ,
    business_type SMALLINT      NOT NULL DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX chat_file_req_idx_chatid_uid_fileid ON chat_file_req (chat_id,uid,file_id);
CREATE INDEX chat_file_req_idx_create_time ON chat_file_req (create_time);

DROP TABLE IF EXISTS chat_file_user;
CREATE TABLE chat_file_user
(
    id                  BIGSERIAL,
    file_id             varchar(64)           DEFAULT NULL ,
    uid                 varchar(128) NOT NULL ,
    file_url            varchar(1024)         DEFAULT NULL ,
    file_name           varchar(128)          DEFAULT NULL ,
    file_size           bigint                DEFAULT NULL ,
    file_pdf_url        varchar(1024)         DEFAULT NULL ,
    create_time         TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    update_time         TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    deleted             SMALLINT      NOT NULL DEFAULT '0' ,
    client_type         SMALLINT      NOT NULL DEFAULT '0' ,
    business_type       SMALLINT      NOT NULL DEFAULT '0' ,
    display             SMALLINT      NOT NULL DEFAULT '0' ,
    file_status         SMALLINT      NOT NULL DEFAULT '1' ,
    file_business_key   varchar(1024)         DEFAULT NULL ,
    extra_link          varchar(1024)         DEFAULT NULL ,
    document_type       SMALLINT               DEFAULT '1' ,
    file_index          int                   DEFAULT NULL ,
    scene_type_id       bigint                DEFAULT NULL ,
    icon                varchar(1024)         DEFAULT NULL ,
    collect_origin_from varchar(1024)         DEFAULT NULL ,
    task_id             varchar(100)          DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX chat_file_user_file_id_IDX ON chat_file_user (file_id);
CREATE INDEX chat_file_user_uid_IDX ON chat_file_user (uid);
CREATE INDEX chat_file_user_create_time_IDX ON chat_file_user (create_time);

DROP TABLE IF EXISTS dataset_file;
CREATE TABLE dataset_file
(
    id            BIGSERIAL ,
    dataset_id    bigint        NOT NULL ,
    dataset_index varchar(255)           DEFAULT NULL ,
    name          varchar(128)  NOT NULL ,
    doc_type      varchar(32)   NOT NULL ,
    doc_url       varchar(2048) NOT NULL ,
    s3_url        varchar(2048)          DEFAULT NULL ,
    para_count    int                    DEFAULT NULL ,
    char_count    int                    DEFAULT NULL ,
    status        SMALLINT       NOT NULL DEFAULT '0' ,
    create_time   TIMESTAMP               DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP               DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX dataset_file_idx_dataset_id ON dataset_file (dataset_id);

DROP TABLE IF EXISTS dataset_info;
CREATE TABLE dataset_info
(
    id          BIGSERIAL ,
    uid         varchar(128) NOT NULL ,
    name        varchar(128) NOT NULL ,
    description varchar(256)          DEFAULT NULL ,
    file_num    int                   DEFAULT NULL ,
    status      SMALLINT      NOT NULL DEFAULT '0' ,
    create_time TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    update_time TIMESTAMP              DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX dataset_info_idx_uid ON dataset_info (uid);
CREATE INDEX dataset_info_idx_create_time ON dataset_info (create_time);

DROP TABLE IF EXISTS extract_knowledge_task;
CREATE TABLE extract_knowledge_task
(
    id          BIGSERIAL ,
    file_id     bigint       DEFAULT NULL ,
    task_id     varchar(64)  DEFAULT NULL ,
    status      int          DEFAULT '0' ,
    reason      text,
    user_id     varchar(128) DEFAULT NULL ,
    create_time timestamp NULL DEFAULT NULL ,
    update_time timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    task_status int          DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS file_directory_tree;
CREATE TABLE file_directory_tree
(
    id          BIGSERIAL ,
    name        varchar(255) DEFAULT NULL ,
    parent_id   bigint       DEFAULT NULL ,
    is_file     SMALLINT DEFAULT '0' ,
    app_id      varchar(10)  DEFAULT NULL ,
    file_id     bigint       DEFAULT NULL ,
    comment     varchar(255) DEFAULT NULL ,
    create_time timestamp NULL DEFAULT NULL ,
    update_time timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    hit_count   int          DEFAULT '0' ,
    status      SMALLINT DEFAULT '0' ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS file_info;
CREATE TABLE file_info
(
    id          BIGSERIAL,
    app_id      varchar(10)        DEFAULT NULL,
    name        varchar(128)       DEFAULT NULL,
    address     varchar(255)       DEFAULT NULL,
    size        bigint             DEFAULT NULL,
    type        varchar(64)        DEFAULT NULL,
    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_id   varchar(255)       DEFAULT NULL,
    status      int                DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS file_info_v2;
CREATE TABLE file_info_v2
(
    id                   BIGSERIAL,
    repo_id              bigint      NOT NULL ,
    uuid                 varchar(64)          DEFAULT NULL,
    uid                  varchar(255)         DEFAULT NULL ,
    name                 varchar(512)         DEFAULT NULL ,
    address              varchar(255)         DEFAULT NULL ,
    size                 bigint               DEFAULT NULL ,
    char_count           bigint               DEFAULT NULL ,
    type                 varchar(64)          DEFAULT NULL ,
    status               int                  DEFAULT NULL ,
    enabled              int                  DEFAULT '0' ,
    slice_config         varchar(500)         DEFAULT NULL ,
    current_slice_config varchar(500)         DEFAULT NULL ,
    pid                  bigint               DEFAULT '-1' ,
    reason               text ,
    create_time          timestamp   NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    update_time          timestamp   NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    source               varchar(64) NOT NULL DEFAULT 'AIUI-RAG2' ,
    space_id             bigint               DEFAULT NULL ,
    last_uuid            varchar(100)         DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS knowledge;
CREATE TABLE knowledge (
                             id varchar(64) NOT NULL ,
                             file_id varchar(64) DEFAULT NULL ,
                             content text,
                             char_count varchar(255) DEFAULT NULL,
                             name varchar(255) DEFAULT NULL,
                             description varchar(512) DEFAULT NULL,
                             enabled BOOLEAN DEFAULT FALSE,
                             source BOOLEAN DEFAULT TRUE,
                             test_hit_count bigint DEFAULT NULL,
                             dialog_hit_count bigint DEFAULT NULL,
                             core_repo_name text,
                             deleted BOOLEAN NOT NULL DEFAULT FALSE,
                             created_at TIMESTAMP NOT NULL,
                             updated_at TIMESTAMP DEFAULT NULL,
                             seq_id BIGSERIAL ,
                             PRIMARY KEY (id));

CREATE UNIQUE INDEX knowledge_uk_seq_id ON knowledge (seq_id);
CREATE INDEX knowledge_flow_id ON knowledge (char_count);
CREATE INDEX knowledge_idx_file_seq ON knowledge (file_id,seq_id);

DROP TABLE IF EXISTS preview_knowledge;
CREATE TABLE preview_knowledge (
                                     id varchar(64) NOT NULL ,
                                     file_id varchar(64) DEFAULT NULL ,
                                     content text,
                                     char_count varchar(255) DEFAULT NULL,
                                     deleted BOOLEAN NOT NULL DEFAULT FALSE,
                                     created_at TIMESTAMP NOT NULL,
                                     updated_at TIMESTAMP DEFAULT NULL,
                                     seq_id BIGSERIAL ,
                                     PRIMARY KEY (id));

CREATE UNIQUE INDEX preview_knowledge_uk_seq_id ON preview_knowledge (seq_id);
CREATE INDEX preview_knowledge_flow_id ON preview_knowledge (char_count);
CREATE INDEX preview_knowledge_idx_file_seq ON preview_knowledge (file_id,seq_id);

DROP TABLE IF EXISTS repo;
CREATE TABLE repo
(
    id             BIGSERIAL ,
    name           varchar(64)          DEFAULT NULL ,
    user_id        varchar(128)         DEFAULT NULL,
    app_id         varchar(20)          DEFAULT NULL,
    outer_repo_id  varchar(50)          DEFAULT NULL,
    core_repo_id   varchar(50)          DEFAULT NULL,
    description    varchar(255)         DEFAULT NULL ,
    icon           varchar(255)         DEFAULT NULL ,
    color          varchar(10)          DEFAULT NULL,
    status         int                  DEFAULT '0' ,
    embedded_model varchar(20)          DEFAULT NULL ,
    index_type     int                  DEFAULT NULL ,
    visibility     int                  DEFAULT '0' ,
    source         int                  DEFAULT '0' ,
    enable_audit   SMALLINT DEFAULT '0' ,
    deleted        SMALLINT DEFAULT '0' ,
    create_time    timestamp NULL DEFAULT NULL ,
    update_time    timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    is_top         BOOLEAN               DEFAULT FALSE,
    tag            varchar(64) NOT NULL DEFAULT 'CBG-RAG' ,
    space_id       bigint               DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS req_knowledge_records;
CREATE TABLE req_knowledge_records
(
    id          BIGSERIAL,
    uid         varchar(128) DEFAULT NULL,
    req_id      bigint        DEFAULT NULL ,
    req_message varchar(8000) DEFAULT NULL ,
    knowledge   varchar(4096) DEFAULT NULL ,
    create_time TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    chat_id     bigint        DEFAULT NULL ,
    PRIMARY KEY (id));

CREATE INDEX req_knowledge_records_idx_uid_req ON req_knowledge_records (uid,req_id);

DROP TABLE IF EXISTS upload_doc_task;
CREATE TABLE upload_doc_task
(
    id              BIGSERIAL ,
    task_id         varchar(64) DEFAULT NULL ,
    extract_task_id varchar(64) DEFAULT NULL ,
    file_id         bigint      DEFAULT NULL ,
    bot_id          bigint      DEFAULT NULL ,
    repo_id         varchar(64) DEFAULT NULL ,
    step            int         DEFAULT NULL ,
    status          int         DEFAULT '0' ,
    reason          text,
    app_id          varchar(60) DEFAULT NULL ,
    create_time     timestamp NULL DEFAULT NULL ,
    update_time     timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id)
);

