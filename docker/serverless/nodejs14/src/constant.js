'use strict';

module.exports = exports = {
    // Environment variables.
    ENV: {
        SERVER_PORT: 'FC_SERVER_PORT', // Server port number.
        SERVER_PATH: 'FC_SERVER_PATH', // Server home directory.
        SERVER_LOG_PATH: 'FC_SERVER_LOG_PATH', // Server application log path.
        SERVER_LOG_LEVEL: 'FC_SERVER_LOG_LEVEL', // Server log level.

        FUNC_CODE_PATH: 'FC_FUNC_CODE_PATH', // Function code path.
        FUNC_LOG_PATH: 'FC_FUNC_LOG_PATH' // Function log path.
    },

    // Environment variables that function can rely on.
    SAFE_ENV: new Set(['FC_FUNC_CODE_PATH']),

    // Invoke request keys.
    INVOKE: {
        REQUEST_ID: 'requestId',
        FUNCTION: 'function',
        EVENT: 'event'
    },

    // Request headers.
    HEADERS: {
        CONTENT_TYPE: 'content-type',
        REQUEST_ID: 'x-fc-request-id',
        ACCESS_KEY_ID: 'x-fc-access-key-id',
        ACCESS_KEY_SECRET: 'x-fc-access-key-secret',
        SECURITY_TOKEN: 'x-fc-security-token',
        FUNCTION_NAME: 'x-fc-function-name',
        FUNCTION_HANDLER: 'x-fc-function-handler',
        FUNCTION_MEMORY: 'x-fc-function-memory',
        FUNCTION_TIMEOUT: 'x-fc-function-timeout',
        FUNCTION_INITIALIZER: 'x-fc-function-initializer',
        FUNCTION_INITIALIZATION_TIMEOUT: 'x-fc-initialization-timeout',
        FUNCTION_PRE_FREEZE_HANDLER: 'x-fc-instance-lifecycle-pre-freeze-handler',
        FUNCTION_PRE_STOP_HANDLER: 'x-fc-instance-lifecycle-pre-stop-handler',

        SERVICE_NAME: 'x-fc-service-name',
        SERVICE_LOG_PROJECT: 'x-fc-service-logproject',
        SERVICE_LOG_STORE: 'x-fc-service-logstore',

        REGION: 'x-fc-region',

        ACCOUNT_ID: 'x-fc-account-id',
        HTTP_PARAMS: 'x-fc-http-params',

        QUALIFIER: 'x-fc-qualifier',
        VERSION_ID: 'x-fc-version-id',

        RETRY_COUNT: 'x-fc-retry-count',
        OPENTRACING_SPAN_CONTEXT: 'x-fc-tracing-opentracing-span-context',
        OPENTRACING_SPAN_BAGGAGES: 'x-fc-tracing-opentracing-span-baggages',
        JAEGER_ENDPOINT: 'x-fc-tracing-jaeger-endpoint'
    },

    LIMIT: {
        ERROR_LOG_LENGTH: 256 * 1024
    },

    // log
    // Start of log tail mark
    LOG_TAIL_START_PREFIX_INVOKE: 'FC Invoke Start RequestId: ',
    LOG_TAIL_START_PREFIX_PREPARE: 'FC Propare Code Start RequestId: ',
    LOG_TAIL_START_PREFIX_INIITALIZE: 'FC Initialize Start RequestId: ',
    LOG_TAIL_START_PREFIX_PRE_STOP: 'FC PreStop Start RequestId: ',
    LOG_TAIL_START_PREFIX_PRE_FREEZE: 'FC PreFreeze Start RequestId: ',
    // End of log tail mark
    LOG_TAIL_END_PREFIX_INVOKE: 'FC Invoke End RequestId: ',
    LOG_TAIL_END_PREFIX_PREPARE: 'FC Prepare Code End RequestId: ',
    LOG_TAIL_END_PREFIX_INITIALIZE: 'FC Initialize End RequestId: ',
    LOG_TAIL_END_PREFIX_PRE_STOP: 'FC PreStop End RequestId: ',
    LOG_TAIL_END_PREFIX_PRE_FREEZE: 'FC PreFreeze End RequestId: ',

    INVOKE_PATH_NAME: '/invoke',
    INITIALIZE_PATH_NAME: '/initialize',
    PRE_FREEZE_PATH_NAME: '/pre-freeze',
    PRE_STOP_PATH_NAME: '/pre-stop'
};
