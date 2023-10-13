/*
 * Logger provides logging methods for application to log debugging messages
 * into a log file. Default logging level is error. You can overwrite logging
 * level by setting the environment variables.
 *
 * Environment variables
 *   - FC_SERVER_LOG_PATH:  Log directory of the application log.
 *   - FC_SERVER_LOG_LEVEL: Logging level.
 */
'use strict';
var path = require('path');
var util = require('util');
var mkdirp = require('mkdirp');
var winston = require('winston');
var dailyRotateFile = require('winston-daily-rotate-file');
var configx = require('./config.js');

const datePattern = '.yyyy-MM-dd-HH';

var newLogger = function(logLevel, logFile) {
    // Create the directory.
    mkdirp.sync(path.dirname(logFile));
    // Create logger.
    return new(winston.Logger)({
        transports: [
            new(dailyRotateFile)({
                level: logLevel,
                filename: logFile,
                datePattern: datePattern,
                json: false,
                handleExceptions: true,
                humanReadableUnhandledException: true,
                timestamp: function() {
                    return new Date().toISOString().
                           replace(/T/, ' ').      // replace T with a space
                           replace(/Z/, '');     // replace Z with ""
                 },
                formatter: function(options) {
                    return util.format('%s [%s] %s %s',
                        options.timestamp(),
                        options.level,
                        options.message ? options.message : '',
                        (options.meta && Object.keys(options.meta).length ? '\n' + JSON.stringify(options.meta, null, 2) : '')
                    );
                }
            })
        ]
    });
};

var cfg = configx.load();
var logLevel = cfg.server.logLevel;
var logFile = cfg.server.logFile;
var logger = newLogger(logLevel, logFile);

// Configurate logger.
exports.config = function(config) {
    if (config && config.server) {
        if (config.server.logLevel) {
            logLevel = config.server.logLevel;
        }
        if (config.server.logFile) {
            logFile = config.server.logFile;
        }

        logger = newLogger(logLevel, logFile);
    }
};

// Return logger.
exports.getLogger = function() {
    return logger;
};
