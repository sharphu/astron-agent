select 'tenant DATABASE initialization started' as "";

CREATE DATABASE tenant;
\connect tenant;

DROP TABLE IF EXISTS tb_app;
CREATE TABLE tb_app (
  update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  registration_time timestamp NULL,
  app_id varchar(32) NOT NULL DEFAULT '',
  app_name varchar(256) NULL,
  dev_id bigint NULL,
  channel_id varchar(128) NULL,
  source varchar(32) NOT NULL DEFAULT '',
  is_disable smallint NULL,
  app_desc varchar(521) NULL,
  is_delete smallint NULL,
  extend varchar(256) NOT NULL DEFAULT '',
  PRIMARY KEY (app_id)
);
CREATE INDEX idx_tb_app_registration_time ON tb_app (registration_time);
CREATE INDEX idx_tb_app_dev_id ON tb_app (dev_id);

INSERT INTO tb_app (update_time, registration_time, app_id, app_name, dev_id, channel_id, source, is_disable, app_desc, is_delete, extend)
VALUES ('2025-09-20 00:00:00', '2025-09-20 00:00:00', '680ab54f', '星辰租户', 1, '0', 'admin', 0, '星辰租户', 0, '');

DROP TABLE IF EXISTS tb_auth;
CREATE TABLE tb_auth (
  update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  registration_time timestamp NULL,
  app_id varchar(32) NOT NULL DEFAULT '',
  api_key varchar(128) NOT NULL DEFAULT '',
  api_secret varchar(128) NULL,
  source bigint NULL,
  is_delete smallint NULL,
  extend varchar(256) NULL,
  PRIMARY KEY (app_id, api_key)
);
CREATE INDEX idx_tb_auth_registration_time ON tb_auth (registration_time);
CREATE INDEX idx_tb_auth_api_key ON tb_auth (api_key);

INSERT INTO tb_auth (update_time, registration_time, app_id, api_key, api_secret, source, is_delete, extend)
VALUES ('2025-09-20 00:00:00', '2025-09-20 00:00:00', '680ab54f', '7b709739e8da44536127a333c7603a83', 'NjhmY2NmM2NkZDE4MDFlNmM5ZjcyZjMy', 0, 0, '');

select 'tenant DATABASE initialization completed' as "";
