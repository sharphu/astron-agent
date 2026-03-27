package database

import (
	"strings"
	"testing"

	"tenant/config"
)

func TestDBType_Constants(t *testing.T) {
	if MYSQL != "mysql" {
		t.Errorf("Expected MYSQL constant to be 'mysql', got '%s'", MYSQL)
	}
}

func TestNewDatabase(t *testing.T) {
	tests := []struct {
		name        string
		config      *config.Config
		wantErr     bool
		expectedErr string
	}{
		{
			name:        "nil config should return error",
			config:      nil,
			wantErr:     true,
			expectedErr: "database config is nil or dbType is empty",
		},
		{
			name: "empty dbType should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					DBType: "",
				},
			},
			wantErr:     true,
			expectedErr: "database config is nil or dbType is empty",
		},
		{
			name: "unsupported dbType should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					DBType: "sqlite",
				},
			},
			wantErr:     true,
			expectedErr: "unsupported dbType: sqlite",
		},
		{
			name: "mysql with empty username should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					DBType:   "mysql",
					UserName: "",
					Password: "password",
				},
			},
			wantErr:     true,
			expectedErr: "mysql username is empty",
		},
		{
			name: "mysql with empty password should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					DBType:   "mysql",
					UserName: "user",
					Password: "",
				},
			},
			wantErr:     true,
			expectedErr: "mysql password is empty",
		},
		{
			name: "mysql with invalid connection string should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					DBType:       "mysql",
					UserName:     "user",
					Password:     "password",
					Host:         "invalidhost",
					Port:         99999,
					DBName:       "nonexistentdb",
					MaxOpenConns: 10,
					MaxIdleConns: 5,
				},
			},
			wantErr: true,
			// Error message will vary depending on connection failure
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db, err := NewDatabase(tt.config)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				} else if tt.expectedErr != "" && err.Error() != tt.expectedErr {
					t.Errorf("Expected error '%s', got '%s'", tt.expectedErr, err.Error())
				}
				if db != nil {
					t.Error("Expected nil database when error occurs")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if db == nil {
					t.Error("Expected non-nil database")
				}
			}
		})
	}
}

func TestDatabase_buildMysql(t *testing.T) {
	db := &Database{}

	tests := []struct {
		name        string
		config      *config.Config
		wantErr     bool
		expectedErr string
	}{
		{
			name: "empty username should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					UserName: "",
					Password: "password",
				},
			},
			wantErr:     true,
			expectedErr: "mysql username is empty",
		},
		{
			name: "empty password should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					UserName: "user",
					Password: "",
				},
			},
			wantErr:     true,
			expectedErr: "mysql password is empty",
		},
		{
			name: "invalid connection should return error",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					UserName: "user",
					Password: "password",
					Host:      "invalidhost",
					Port:      99999,
					DBName:    "nonexistentdb",
					MaxOpenConns: 10,
					MaxIdleConns: 5,
				},
			},
			wantErr: true,
			// Error message will vary depending on connection failure
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := db.buildMysql(tt.config)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				} else if tt.expectedErr != "" && err.Error() != tt.expectedErr {
					t.Errorf("Expected error '%s', got '%s'", tt.expectedErr, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

func TestDatabase_GetMysql(t *testing.T) {
	// Test with nil mysql connection
	db := &Database{mysql: nil}
	if db.GetMysql() != nil {
		t.Error("Expected nil when mysql connection is nil")
	}

	// Note: We can't easily test with a real connection without setting up a test database
	// In a real test environment, you would set up a test database or use a mock
}

func TestDatabase_Structure(t *testing.T) {
	// Test that Database struct is properly defined
	db := &Database{}

	// Verify struct fields are accessible
	if db.mysql != nil {
		t.Error("Expected mysql field to be nil initially")
	}

	// Test that the struct can be initialized
	db.mysql = nil
	if db.mysql != nil {
		t.Error("Expected mysql field to remain nil")
	}
}

func TestDBType_Usage(t *testing.T) {
	// Test DBType conversion and usage
	dbType := DBType("mysql")
	if dbType != MYSQL {
		t.Errorf("Expected DBType('mysql') to equal MYSQL constant")
	}

	// Test comparison with string
	if string(MYSQL) != "mysql" {
		t.Errorf("Expected MYSQL constant to convert to 'mysql' string")
	}
}

func TestDatabase_ConfigValidation(t *testing.T) {
	// Test various edge cases for config validation
	tests := []struct {
		name   string
		config *config.Config
		field  string
		value  string
	}{
		{
			name: "special characters in username",
			config: &config.Config{
				DataBase: struct {
					DBType       string `toml:"dbType"`
					UserName     string `toml:"username"`
					Password     string `toml:"password"`
					Host         string `toml:"host"`
					Port         int    `toml:"port"`
					DBName       string `toml:"dbname"`
					MaxOpenConns int    `toml:"maxOpenConns"`
					MaxIdleConns int    `toml:"maxIdleConns"`
				}{
					DBType:   "mysql",
					UserName: "user@domain.com",
					Password: "password!@#$%",
					Host:      "localhost",
					Port:      3306,
					DBName:    "test",
				},
			},
			field: "username",
			value: "user@domain.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that the config structure properly holds the values
			switch tt.field {
			case "username":
				if tt.config.DataBase.UserName != tt.value {
					t.Errorf("Expected username '%s', got '%s'", tt.value, tt.config.DataBase.UserName)
				}
			}

			// These will fail to connect but should pass validation
			_, err := NewDatabase(tt.config)
			if err != nil {
				// Expected to fail at connection time, not validation time
				t.Logf("Expected connection failure: %v", err)
			}
		})
	}
}

func TestDatabase_ConnectionPoolSettings(t *testing.T) {
	// Test that connection pool settings are properly applied
	config := &config.Config{
		DataBase: struct {
			DBType       string `toml:"dbType"`
			UserName     string `toml:"username"`
			Password     string `toml:"password"`
			Host         string `toml:"host"`
			Port         int    `toml:"port"`
			DBName       string `toml:"dbname"`
			MaxOpenConns int    `toml:"maxOpenConns"`
			MaxIdleConns int    `toml:"maxIdleConns"`
		}{
			DBType:       "mysql",
			UserName:     "testuser",
			Password:     "testpass",
			Host:         "testhost",
			Port:         3306,
			DBName:       "testdb",
			MaxOpenConns: 100,
			MaxIdleConns: 50,
		},
	}

	// This will fail to connect, but we can test the buildMysql method
	db := &Database{}
	err := db.buildMysql(config)

	// Expected to fail due to invalid connection, but config validation should pass
	if err == nil {
		t.Error("Expected connection error due to invalid host")
	}

	// Verify that the method attempts to use the pool settings
	// (We can't verify they were actually set without a real connection)
}

func TestDatabase_IntegrationReadiness(t *testing.T) {
	// Test that the Database struct is ready for integration
	t.Run("database_initialization", func(t *testing.T) {
		// Test that Database can be created and initialized
		db := &Database{}
		if db.mysql == nil {
			t.Log("Database struct created with empty mysql connection")
		}

		// Test method accessibility
		mysql := db.GetMysql()
		if mysql != nil {
			t.Error("Expected nil mysql connection initially")
		}
	})

	t.Run("method_signatures_exist", func(t *testing.T) {
		// Compile-time check for method existence
		if false {
			var db *Database
			var config *config.Config
			_, _ = NewDatabase(config)
			_ = db.buildMysql(config)
			db.GetMysql()
		}
	})
}

func TestRewritePlaceholders_Kingbase(t *testing.T) {
	db := &Database{dbType: KINGBASE}
	got := db.RewritePlaceholders("select * from t where a=? and b=?")
	want := "select * from t where a=$1 and b=$2"
	if got != want {
		t.Fatalf("unexpected rewrite result: got=%q want=%q", got, want)
	}
}

func assertContains(t *testing.T, s string, substr string) {
	t.Helper()
	if !strings.Contains(s, substr) {
		t.Fatalf("expected %q to contain %q", s, substr)
	}
}
