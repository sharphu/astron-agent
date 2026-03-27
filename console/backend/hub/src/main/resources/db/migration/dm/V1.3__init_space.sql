-- Migration script for init_space

DROP TABLE IF EXISTS agent_space;
CREATE TABLE agent_space
(
    id            BIGINT IDENTITY(1,1),
    name          varchar(50) NOT NULL ,
    description   varchar(2000) DEFAULT NULL ,
    avatar_url    varchar(1024) DEFAULT NULL ,
    uid           varchar(128)  DEFAULT NULL ,
    enterprise_id bigint        DEFAULT NULL ,
    type          SMALLINT       DEFAULT NULL ,
    create_time   TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    deleted       SMALLINT       DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX agent_space_uid_key ON agent_space (uid);
CREATE INDEX agent_space_enterprise_id_key ON agent_space (enterprise_id);
CREATE INDEX agent_space_space_name ON agent_space (name);

DROP TABLE IF EXISTS agent_space_permission;
CREATE TABLE agent_space_permission
(
    id                BIGINT IDENTITY(1,1),
    module            varchar(50)  DEFAULT NULL ,
    point             varchar(50)  DEFAULT NULL ,
    description       varchar(255) DEFAULT NULL ,
    permission_key    varchar(128)  DEFAULT NULL ,
    owner             SMALLINT NOT NULL ,
    admin             SMALLINT NOT NULL ,
    member            SMALLINT NOT NULL ,
    available_expired SMALLINT NOT NULL ,
    create_time       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX agent_space_permission_key_uni_key ON agent_space_permission (permission_key);

DROP TABLE IF EXISTS agent_space_user;
CREATE TABLE agent_space_user
(
    id              BIGINT IDENTITY(1,1),
    space_id        bigint       NOT NULL ,
    uid             varchar(128) NOT NULL ,
    nickname        varchar(64) DEFAULT NULL ,
    role            SMALLINT      NOT NULL ,
    last_visit_time TIMESTAMP    DEFAULT NULL ,
    create_time     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    update_time     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX agent_space_user_space_id_uid_uni_key ON agent_space_user (space_id,uid);
CREATE INDEX agent_space_user_space_user_uid_key ON agent_space_user (uid);

