/*
 * Load fc environment variables. Environment are defined in
 * constant.js
 */
'use strict';
var constant = require('./constant.js');

/*
 * Return an object contains fc environment variables.
 */
exports.load = function() {
    var env = {};
    for (let i in constant.ENV) {
        var name = constant.ENV[i];
        env[name] = process.env[name];
    }
    return env;
};
