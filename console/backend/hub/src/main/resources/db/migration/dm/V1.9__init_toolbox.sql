-- Migration script for init_toolbox

DROP TABLE IF EXISTS tool_box;
CREATE TABLE tool_box
(
    id              BIGINT IDENTITY(1,1) ,
    tool_id         varchar(30)      DEFAULT NULL ,
    name            varchar(64)      DEFAULT NULL ,
    description     varchar(255)     DEFAULT NULL ,
    icon            varchar(255)     DEFAULT NULL ,
    user_id         varchar(256)     DEFAULT NULL ,
    app_id          varchar(60)      DEFAULT NULL ,
    end_point       text ,
    method          varchar(255)     DEFAULT NULL ,
    web_schema      TEXT ,
    schema          TEXT ,
    visibility      int              DEFAULT '0' ,
    deleted         SMALLINT DEFAULT '0' ,
    create_time     timestamp NULL DEFAULT NULL ,
    update_time     timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    is_public       BOOLEAN           DEFAULT FALSE,
    favorite_count  int              DEFAULT '0' ,
    usage_count     int              DEFAULT '0' ,
    tool_tag        varchar(255)     DEFAULT NULL,
    operation_id    varchar(255)     DEFAULT NULL,
    creation_method SMALLINT          DEFAULT '0',
    auth_type       SMALLINT          DEFAULT '0',
    auth_info       varchar(1024)    DEFAULT NULL,
    top             int              DEFAULT '0',
    source          SMALLINT          DEFAULT '1',
    display_source  varchar(16)      DEFAULT '1,2',
    avatar_color    varchar(255)     DEFAULT NULL,
    status          SMALLINT NOT NULL DEFAULT '1' ,
    version         varchar(100)     DEFAULT NULL,
    temporary_data  TEXT ,
    space_id        bigint           DEFAULT NULL ,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS tool_box_copy;
CREATE TABLE tool_box_copy
(
    id              BIGINT IDENTITY(1,1) ,
    tool_id         varchar(30)   DEFAULT NULL ,
    name            varchar(64)   DEFAULT NULL ,
    description     varchar(255)  DEFAULT NULL ,
    icon            varchar(255)  DEFAULT NULL ,
    user_id         varchar(20)   DEFAULT NULL ,
    app_id          varchar(60)   DEFAULT NULL ,
    end_point       text ,
    method          varchar(255)  DEFAULT NULL ,
    web_schema      TEXT ,
    schema          TEXT ,
    visibility      int           DEFAULT '0' ,
    deleted         SMALLINT DEFAULT '0' ,
    create_time     timestamp NULL DEFAULT NULL ,
    update_time     timestamp NULL DEFAULT CURRENT_TIMESTAMP ,
    is_public       BOOLEAN        DEFAULT FALSE,
    favorite_count  int           DEFAULT '0' ,
    usage_count     int           DEFAULT '0' ,
    tool_tag        varchar(255)  DEFAULT NULL,
    operation_id    varchar(255)  DEFAULT NULL,
    creation_method SMALLINT       DEFAULT '0',
    auth_type       SMALLINT       DEFAULT '0',
    auth_info       varchar(1024) DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS tool_box_feedback;
CREATE TABLE tool_box_feedback
(
    id          BIGINT IDENTITY(1,1),
    user_id     varchar(100) NOT NULL ,
    tool_id     varchar(100)          DEFAULT NULL ,
    name        varchar(100)          DEFAULT NULL ,
    remark      varchar(1000)         DEFAULT NULL ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS tool_box_heat_value;
CREATE TABLE tool_box_heat_value
(
    id         INT IDENTITY(1,1),
    tool_name  varchar(100) DEFAULT NULL,
    heat_value int          DEFAULT NULL,
    PRIMARY KEY (id)
);

DROP TABLE IF EXISTS tool_box_operate_history;
CREATE TABLE tool_box_operate_history
(
    id          BIGINT IDENTITY(1,1),
    tool_id     varchar(100) NOT NULL ,
    uid         varchar(100) NOT NULL,
    type        SMALLINT      NOT NULL ,
    create_time TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);

