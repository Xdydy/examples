'use strict';
var util = require('util');
var path = require('path');
var constant = require('./constant.js');

var codePath = null;

/*
 * Configure sanitizer.
 */
exports.config = function(config) {
    if (config && config.func && config.func.codePath) {
        codePath = config.func.codePath;
    }
};

/*
 * Remove system information in stack trace.
 */
exports.washErrorStack = function() {
    var fcStackTrace = function(error, structuredStackTrace) {
        // Check if the error is caused by user function code.
        var funcError = error.shouldSanitize;
        if (!funcError) {
            for (let i in structuredStackTrace) {
                var st = structuredStackTrace[i];
                if (codePath && st.getFileName() && st.getFileName().includes(codePath)) {
                    funcError = true;
                    break;
                }
            }
        }

        if (funcError) {
            var stacks = [];
            var dir = path.parse(__dirname).dir;
            stacks.push(util.format('%s: %s', error.name, error.message));
            for (let i in structuredStackTrace) {
                var st = structuredStackTrace[i];
                if (st.getFileName() && st.getFileName().includes(dir)) {
                    // Stack trace contains server information.
                    break;
                } else {
                    if (st.getFunctionName()) {
                        stacks.push(util.format('    at %s (%s:%s:%s)',
                            st.getFunctionName(),
                            st.getFileName(),
                            st.getLineNumber(),
                            st.getColumnNumber()
                        ));
                    } else {
                        if (st.getFileName()) {
                            stacks.push(util.format('    at (%s:%s:%s)',
                                st.getFileName(),
                                st.getLineNumber(),
                                st.getColumnNumber()
                            ));
                        } else {
                            stacks.push('    at (<anonymous>)');
                        }
                    }
                }
            }

            var fcStack = stacks.join('\n');
            error.stack = fcStack;
            return fcStack;
        } else {
            return error.stack;
        }
    };

    // Customize stack traces
    // See https://github.com/v8/v8/wiki/Stack%20Trace%20API.
    Error.prepareStackTrace = function(error, structuredStackTrace) {
        return fcStackTrace(error, structuredStackTrace);
    };
};

/*
 * Remove fc environment variables.
 */
exports.washEnv = function() {
    for (let i in process.env) {
        if (i.startsWith('FC_') && !constant.SAFE_ENV.has(i)){
            delete process.env[i];
        }
    }
};
