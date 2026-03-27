package com.iflytek.astron.console.toolkit.util.database.dialect;

import java.util.Locale;

public enum DbDialectType {

    MYSQL,
    POSTGRES,
    KINGBASE;

    public static DbDialectType fromString(String value) {
        if (value == null || value.isBlank()) {
            return MYSQL;
        }
        switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "mysql":
                return MYSQL;
            case "postgres":
            case "postgresql":
            case "pg":
                return POSTGRES;
            case "kingbase":
                return KINGBASE;
            default:
                throw new IllegalArgumentException("Unsupported DB dialect: " + value);
        }
    }
}
