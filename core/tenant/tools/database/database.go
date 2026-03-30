package database

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"tenant/config"

	_ "github.com/go-sql-driver/mysql"
	_ "gitea.com/kingbase/gokb"
)

type DBType string

const (
	MYSQL    DBType = "mysql"
	KINGBASE DBType = "kingbase"
)

var pgFamily = map[DBType]struct{}{
	KINGBASE:     {},
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
	case KINGBASE:
		err := db.buildKingbase(conf)
		if err != nil {
			return nil, err
		}
		db.dbType = KINGBASE
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
	if strings.TrimSpace(conf.DataBase.Host) == "" {
		return errors.New("mysql host is empty")
	}
	if conf.DataBase.Port == 0 {
		return errors.New("mysql port is empty")
	}
	if strings.TrimSpace(conf.DataBase.DBName) == "" {
		return errors.New("mysql dbname is empty")
	}
	hostPortDbname := fmt.Sprintf("(%s:%d)/%s", strings.TrimSpace(conf.DataBase.Host), conf.DataBase.Port, strings.TrimSpace(conf.DataBase.DBName))
	dsn := fmt.Sprintf("%s:%s@tcp%s", conf.DataBase.UserName, conf.DataBase.Password, hostPortDbname)
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
	if len(strings.TrimSpace(conf.DataBase.Host)) == 0 {
		return errors.New("kingbase host is empty")
	}
	if conf.DataBase.Port == 0 {
		return errors.New("kingbase port is empty")
	}
	if len(strings.TrimSpace(conf.DataBase.DBName)) == 0 {
		return errors.New("kingbase dbname is empty")
	}
	connStr := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		strings.TrimSpace(conf.DataBase.Host),
		conf.DataBase.Port,
		strings.TrimSpace(conf.DataBase.UserName),
		strings.TrimSpace(conf.DataBase.Password),
		strings.TrimSpace(conf.DataBase.DBName),
		"disable",
	)
	client, err := sql.Open("kingbase", connStr)
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
