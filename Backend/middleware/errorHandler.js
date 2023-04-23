const { logEvents } = require('./logger')

// Handles errors in the code and returns the response/error code and message and logs to file
const errorHandler = (err, req, res, next) => {
    logEvents(`${err.name}: ${err.message}\t${req.method}\t${req.url}\t${req.headers.origin}`, 'errLog.log')
    console.log(err.stack)

    const status = res.statusCode ? res.statusCode : 500 // server error 
    res.status(status)
    res.json({ message: err.message })
}

module.exports = errorHandler 