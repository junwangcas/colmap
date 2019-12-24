//
// Created by witck on 18-12-14.
//
#pragma once

#include <util/logging_app.h>
#include <string>
#include <fstream>
#include <map>
#include <regex>

class AppConfig {
public:
    explicit AppConfig(const std::string& path) {
        load_config(path);
    }

public:

    const std::map<std::string, std::string>& get_config_value() {
        return config_value_map;
    }

    const std::string get(const std::string& key) {
        if (config_value_map.find(key) == config_value_map.end()) {
            ERROR() << "config key not found, key: " << key;
            throw std::runtime_error("key_not_found");
        }

        return config_value_map[key];
    }

    const std::string operator[](const std::string& key) {
        return get(key);
    }

    const double get_double(const std::string& key) {
        std::string value = get(key);

        if (!is_float(value)) {
            ERROR() << "config value of key " << key << " is not double format; value: " << value;
            throw std::runtime_error("incorrect_format");
        }

        return atof(value.c_str());
    }

    const int get_int(const std::string& key) {
        std::string value = get(key);

        if (!is_int(value)) {
            ERROR() << "config value of key " << key << " is not int format; value: " << value;
            throw std::runtime_error("key not found");
        }

        return int(atol(value.c_str()));
    }

    const long get_long(const std::string& key) {
        std::string value = get(key);
        return atol(value.c_str());
    }


private:
    void load_config(const std::string& path) {
        std::ifstream ifs(path);

        if (!ifs.is_open()) {
            ERROR() << "config file open failed, path: " << path;
            throw std::runtime_error("open failed");
        }

        std::string line;

        while (getline(ifs, line)) {
            trim(line);
            std::regex reg("(.+) *: *([^#\r\n]*)(#.*)*");
            std::regex comments_reg("^#.*");

            if (line.empty() || std::regex_match(line, comments_reg)) {
                continue;
            }

            if (!std::regex_match(line, reg)) {
                ERROR() << "invalid config line: " << line;
                throw std::runtime_error("invalid_line");
            }

            std::smatch sm;
            std::regex_match(line, sm, reg);
            std::string key = sm[1];
            std::string value = sm[2];
            trim(key);
            trim(value);

            if (config_value_map.find(key) != config_value_map.end()) {
                ERROR() << "key '" << key << "' is not unique";
                throw std::runtime_error("key duplicate");
            }

            config_value_map[key] = value;

        }
    }

    bool is_match(const std::string& str, const std::string& reg_str) {
        std::regex reg(reg_str);
        return std::regex_match(str, reg);
    }

    bool is_int(const std::string& str) {
        return is_match(str, "\\d+");
    }

    bool is_float(const std::string& str) {
        return is_match(str, R"(-?\d+(\.\d+)?)");
    }

    std::string& trim(std::string& s) {
        if (s.empty()) {
            return s;
        }

        s.erase(0, s.find_first_not_of(' '));
        s.erase(s.find_last_not_of(' ') + 1);
        return s;
    }


private:
    std::map<std::string, std::string> config_value_map;
};
