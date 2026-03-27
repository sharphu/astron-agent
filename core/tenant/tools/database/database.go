package database

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"tenant/config"

	_ "github.com/go-sql-driver/mysql"
	_ "github.com/lib/pq"
	_ "gitee.com/chunanyong/dm"
)

type DBType string

const (
	MYSQL    DBType = "mysql"
	KINGBASE DBType = "kingbase"
	DM       DBType = "dm"
)

var pgFamily = map[DBType]struct{}{
	KINGBASE:     {},
	"postgres":   {},
	"postgresql": {},
	"pg":         {},
}

type Database struct {
	mysql  *sql.DB
	dbType DBType
}

func NewDatabase(conf *config.Config) (*Database, error) {
	if conf == nil || len(conf.DataBase.DBType) == 0 {
		return nil, errors.New("database config is nil or dbType is empty")
	}
	dbType := DBType(strings.ToLower(strings.TrimSpace(conf.DataBase.DBType)))
	db := &Database{}
	switch dbType {
	case MYSQL:
		err := db.buildMysql(conf)
		if err != nil {
			return nil, err
		}
		db.dbType = MYSQL
		return db, nil
	case KINGBASE, "postgresql", "postgres", "pg":
		err := db.buildKingbase(conf)
		if err != nil {
			return nil, err
		}
		db.dbType = KINGBASE
		return db, nil
	case DM:
		err := db.buildDM(conf)
		if err != nil {
			return nil, err
		}
		db.dbType = DM
		return db, nil
	default:
		return nil, fmt.Errorf("unsupported dbType: %s", conf.DataBase.DBType)
	}
}

func (db *Database) buildMysql(conf *config.Config) error {
	if len(conf.DataBase.UserName) == 0 {
		return errors.New("mysql username is empty")
	}

	if len(conf.DataBase.Password) == 0 {
		return errors.New("mysql password is empty")
	}

	if len(conf.DataBase.Url) == 0 {
		return errors.New("mysql url is empty")
	}
	dsn := fmt.Sprintf("%s:%s@tcp%s", conf.DataBase.UserName, conf.DataBase.Password, conf.DataBase.Url)
	client, err := sql.Open("mysql", dsn)
	if err != nil {
		return err
	}
	client.SetMaxOpenConns(conf.DataBase.MaxOpenConns)
	client.SetMaxIdleConns(conf.DataBase.MaxIdleConns)
	err = client.Ping()
	if err != nil {
		return err
	}
	db.mysql = client
	return nil
}

func (db *Database) buildKingbase(conf *config.Config) error {
	if len(conf.DataBase.UserName) == 0 {
		return errors.New("kingbase username is empty")
	}
	if len(conf.DataBase.Password) == 0 {
		return errors.New("kingbase password is empty")
	}
	if len(conf.DataBase.Url) == 0 {
		return errors.New("kingbase url is empty")
	}
	dsn := conf.DataBase.Url
	if !strings.Contains(dsn, "user=") {
		dsn = fmt.Sprintf("%s user=%s password=%s", dsn, conf.DataBase.UserName, conf.DataBase.Password)
	}
	client, err := sql.Open("postgres", dsn)
	if err != nil {
		return err
	}
	client.SetMaxOpenConns(conf.DataBase.MaxOpenConns)
	client.SetMaxIdleConns(conf.DataBase.MaxIdleConns)
	err = client.Ping()
	if err != nil {
		return err
	}
	db.mysql = client
	return nil
}

func (db *Database) buildDM(conf *config.Config) error {
	if len(conf.DataBase.UserName) == 0 {
		return errors.New("dm username is empty")
	}
	if len(conf.DataBase.Password) == 0 {
		return errors.New("dm password is empty")
	}
	if len(conf.DataBase.Url) == 0 {
		return errors.New("dm url is empty")
	}
	dsn := fmt.Sprintf("dm://%s:%s@%s", conf.DataBase.UserName, conf.DataBase.Password, conf.DataBase.Url)
	client, err := sql.Open("dm", dsn)
	if err != nil {
		return err
	}
	client.SetMaxOpenConns(conf.DataBase.MaxOpenConns)
	client.SetMaxIdleConns(conf.DataBase.MaxIdleConns)
	err = client.Ping()
	if err != nil {
		return err
	}
	db.mysql = client
	return nil
}

func (db *Database) GetMysql() *sql.DB {
	return db.mysql
}

func (db *Database) GetDB() *sql.DB {
	return db.mysql
}

func (db *Database) IsKingbase() bool {
	if db == nil {
		return false
	}
	_, ok := pgFamily[db.dbType]
	return ok
}

func (db *Database) RewritePlaceholders(sqlText string) string {
	if !db.IsKingbase() {
		return sqlText
	}
	var builder strings.Builder
	builder.Grow(len(sqlText) + 16)
	index := 1
	for _, ch := range sqlText {
		if ch == '?' {
			builder.WriteString(fmt.Sprintf("$%d", index))
			index++
			continue
		}
		builder.WriteRune(ch)
	}
	return builder.String()
}
