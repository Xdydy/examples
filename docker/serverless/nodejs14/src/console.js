/*
 * This file exports a redirect() to redirect console logging data to log
 * file. By calling redirect(), following console logging method will be
 * overwritten.
 *
 * Overwritten methods
 *   - console.log
 *   - console.info
 *   - console.warn
 *   - console.error
 *
 * Once console logging data is redirected. Logging data will be saved in
 * a local log file. You can change the log file location by setting the
 * environment variable.
 *
 * Environment variables
 *   - FC_FUNC_LOG_PATH:    The directory of the log file.
 */
'use strict';

var util = require('util');
var winston = require('winston');
var constant = require('./constant.js');
var slogger = require('./logger.js');

var logLevel = 'silly';

console.setLogLevel = function(lv) {
    /*
    const levels = {
      error: 0,
      warn: 1,
      info: 2,
      verbose: 3,
      debug: 4,
      silly: 5
    }
    */
    var formatLv = lv.toLowerCase();
    var lvs = ['error','warn','info','verbose','debug','silly'];
    if(lvs.indexOf(formatLv) == -1){
        return;
    }
    logLevel = formatLv;
    consoleRedirect();
};

var consoleRedirect = function(){
     // Logger for user function.
    var logger = new(winston.Logger)({
        transports: [
            new(winston.transports.Console)({
                level: logLevel,
                json: false,
                timestamp: function() {
                    return new Date().toISOString();
                },
                formatter: function(options) {
                    return util.format('%s %s [%s] %s',
                        options.timestamp(),
                        process._fc && process._fc.requestId ? process._fc.requestId : '',
                        options.level,
                        options.message ? options.message : ''
                    );
                }
            })
        ]
    });

    var log = function(level, data) {
        var msg = util.format.apply(this, data);
        logger.log(level, msg);
    };
    console.log = function() {
        log('verbose', arguments);
    };
    console.info = function() {
        log('info', arguments);
    };
    console.warn = function() {
        log('warn', arguments);
    };
    console.error = function() {
        log('error', arguments);
    };
    console.debug = function() {
       log('debug', arguments);
    };
}

// Configurate logger.
exports.config = function(config) {
    if (config && config.func) {
        if (config.func.logLevel) {
            logLevel = config.func.logLevel;
        }
    }
    // Overwrite console methods to redirect data to logger.
    exports.redirect = function() {
       consoleRedirect();
    };
};

// The max length of an auto log error message.
var maxErrLen = constant.LIMIT.ERROR_LOG_LENGTH;

// Log an error message and cap at max lenght.
exports.errorCap = function(msg) {
    var newMsg = msg.substring(0, maxErrLen) + (msg.length > maxErrLen ? '... Truncated by FunctionCompute' : '');
    console.error(newMsg);
    slogger.getLogger().error(newMsg);
};
