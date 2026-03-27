-- Migration script for init_enterprise

DROP TABLE IF EXISTS agent_enterprise;
CREATE TABLE agent_enterprise
(
    id           BIGINT IDENTITY(1,1),
    uid          varchar(128)  DEFAULT NULL ,
    name         varchar(50)   DEFAULT NULL ,
    logo_url     varchar(1024) DEFAULT NULL ,
    avatar_url   varchar(1024) NOT NULL ,
    org_id       bigint        DEFAULT NULL ,
    service_type SMALLINT       DEFAULT NULL ,
    create_time  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    expire_time  TIMESTAMP      DEFAULT NULL ,
    update_time  TIMESTAMP      DEFAULT CURRENT_TIMESTAMP ,
    deleted      SMALLINT       DEFAULT '0' ,
    PRIMARY KEY (id));

CREATE INDEX agent_enterprise_enterprise_name_key ON agent_enterprise (name);
CREATE INDEX agent_enterprise_enterprise_uid_key ON agent_enterprise (uid);

DROP TABLE IF EXISTS agent_enterprise_permission;
CREATE TABLE agent_enterprise_permission
(
    id                BIGINT IDENTITY(1,1),
    module            varchar(50)  DEFAULT NULL ,
    description       varchar(255) DEFAULT NULL ,
    permission_key    varchar(128)  DEFAULT NULL ,
    officer           SMALLINT NOT NULL ,
    governor          SMALLINT NOT NULL ,
    staff             SMALLINT NOT NULL ,
    available_expired SMALLINT NOT NULL ,
    create_time       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE INDEX agent_enterprise_permission_key_uni_key ON agent_enterprise_permission (permission_key);

DROP TABLE IF EXISTS agent_enterprise_user;
CREATE TABLE agent_enterprise_user
(
    id            BIGINT IDENTITY(1,1),
    enterprise_id bigint       DEFAULT NULL ,
    uid           varchar(128) DEFAULT NULL ,
    nickname      varchar(64)  DEFAULT NULL ,
    role          SMALLINT      DEFAULT NULL ,
    create_time   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    update_time   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ,
    PRIMARY KEY (id));

CREATE UNIQUE INDEX agent_enterprise_user_enterprise_id_uid_uni_key ON agent_enterprise_user (enterprise_id,uid);
CREATE INDEX agent_enterprise_user_enterprise_user_uid_key ON agent_enterprise_user (uid);

