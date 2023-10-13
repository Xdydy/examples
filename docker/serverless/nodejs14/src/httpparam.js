'use strict';

var bufferBuilder = require('./buffer_builder.js');
var events = require('events');
var stream = require('stream');

const req = Symbol('req');
const errHeaderSent = new Error('can not set headers after they are sent');
const errHeaderKeyType = new TypeError('headerKey should be a `string`');
const errHeadValueType = new TypeError('headerValue should be a `string` or a object with `string` type key-value pair');

// see https://nodejs.org/dist/latest-v10.x/docs/api/http.html#http_message_headers
const headerKeyUniqTypeMap = {
    age:true,
    authorization:true,
    'content-length':true,
    'content-type':true,
    etag:true,
    expires:true,
    from:true,
    host:true,
    'if-modified-since':true,
    'if-unmodified-since':true,
    'last-modified':true,
    location:true,
    'max-forwards':true,
    'proxy-authorization':true,
    'referer':true,
    'retry-after':true,
    'user-agent':true
}

const headerKeyArrayTypeMap = {
    'set-cookie':true
}

function parseRequestHeader(evt, json) {
    evt.headers = {};
    var putHeader = function(k, v) {
        var lowerKey = k.toLowerCase();
        var headerValue = evt.headers[lowerKey];
        // should be the first value for uniq key
        if (headerValue && headerKeyUniqTypeMap[lowerKey]) {
            return false;
        }
        // as a array
        if (headerKeyArrayTypeMap[lowerKey]) {
            if (!headerValue) {
                headerValue = [];
            }
            headerValue.push(v);
            evt.headers[lowerKey] = headerValue;
            return true;
        }
        // not exists.
        if (!headerValue) {
            evt.headers[lowerKey] = v;
            return true;
        }
        // join by spliter `, `
        // see https://stackoverflow.com/questions/3096888/standard-for-adding-multiple-values-of-a-single-http-header-to-a-request-or-resp
        evt.headers[lowerKey] = `${headerValue}, ${v}`;
        return true;
    }
    if (json.headersMap && Object.keys(json.headersMap).length > 0) {
        for (var k in json.headersMap) {
            var values = json.headersMap[k];
            for (var i = 0; i < values.length; i++) {
                var v = values[i];
                putHeader(k, v);
            };
        };
    } else {
        for (var k in json.headers) {
            var v = json.headers[k];
            putHeader(k, v);
        };
    }
}

function Request(rawReq, json) {
    var evt = new events.EventEmitter();
    evt[req] = rawReq;

    json = json || {};
    evt.method = json.method || '';
    evt.clientIP = json.clientIP || '';
    evt.url = json.requestURI || '';
    evt.path = json.path || '/';
    evt.queries = {};
    evt.headers = {};

    parseRequestHeader(evt, json);

    if (json.queriesMap && Object.keys(json.queriesMap).length > 0) {
        // fill into queries
        for (var k in json.queriesMap) {
            var v = json.queriesMap[k];
            if (v.length > 1) {
                evt.queries[k] = v;
            } else {
                evt.queries[k] = v[0];
            }
        };
    } else {
        for (var k in json.queries) {
            evt.queries[k] = json.queries[k];
        };
    }

    evt.getHeader = function (headerKey) {
        var lowerKey = headerKey.toLowerCase();
        return evt.headers[lowerKey];
    };

    rawReq.on('data', function(data) {
        evt.emit('data', data);
    });

    rawReq.on('end', function() {
        evt.emit('end');
    });

    return evt;
}

function Response(cb) {
    this._cb = cb;
    this.statusCode = 200;
    this.headers = {};
    this.headersMap = {};
    this.body = bufferBuilder.from('');
    this._headersSent = false;
}

Response.prototype.send = function(buffer) {
    if (this._headersSent) {
        throw new Error('can not send multi times');
    }
    if (typeof buffer === 'string') {
        buffer = bufferBuilder.from(buffer);
    }
    if (!(buffer instanceof Buffer) && !(buffer instanceof stream.Readable)) {
        throw new TypeError('buffer should be a `Buffer` or a `string` or a `stream.Readable`');
    }
    this.body = buffer;
    this._headersSent = true;
    this._cb(this);
}

var checkValues = function(value) {
    if (typeof value === 'string') {
        return true;
    } else if (!Array.isArray(value)){
        return false;
    }
    for (var i = 0; i < value.length; i++) {
        var v = value[i];
        if (typeof v !== 'string') {
            return false;
        }
    }
    return true;
}

// setHeader('key', 'value')
// setHeader('key', ['v1', 'v2'])
Response.prototype.setHeader = function(headerKey, headerValue) {
    if (this._headersSent) {
        throw errHeaderSent;
    }
    if (typeof headerKey !== 'string') {
        throw errHeaderKeyType;
    }
    if (!checkValues(headerValue)) {
        throw errHeadValueType;
    }

    var values = [];
    this.headersMap[headerKey] = values;

    if (typeof headerValue === 'string') {
        values.push(headerValue);
        this.headers[headerKey] = headerValue;
    } else if (Array.isArray(headerValue)) {
        for (var i = 0; i < headerValue.length; i++) {
            values.push(headerValue[i]);
        }
        // do not handle this.headers
    }
    return this;
}

Response.prototype.deleteHeader = function(headerKey) {
    if (this._headersSent) {
        throw new Error("can not set headers after they are sent")
    }
    if (typeof headerKey !== 'string') {
        throw new TypeError('headerKey should be a `string`');
    }
    delete this.headers[headerKey];
    delete this.headersMap[headerKey];
}

Response.prototype.hasHeader = function(headerKey) {
    return this.headersMap[headerKey] !== undefined
}

Response.prototype.setStatusCode = function(statusCode) {
    if (this._headersSent) {
        throw new Error("can not set statusCode after it is sent")
    }
    if (!Number.isInteger(statusCode)) {
        throw new TypeError('statusCode should be a `integer`');
    }
    this.statusCode = statusCode;
}

exports.Response = Response;

// do not export to user.
exports.dumpHTTPParams = function(resp) {
    var json = {
        status: resp.statusCode,
        headers: resp.headers,
        headersMap: resp.headersMap,
    }
    var httpParams = (bufferBuilder.from(JSON.stringify(json))).toString('base64');
    return httpParams
}

exports.parseRequest = function(rawReq, httpParams) {
    var jsonStr = bufferBuilder.from(httpParams, 'base64');
    if (!jsonStr || jsonStr == '') {
        return new Request(rawReq)
    }
    var json = JSON.parse(jsonStr);
    return new Request(rawReq, json);
}
