const express = require('express')
const app = express()
const path = require('path')
const fileUpload = require('express-fileupload')
const { logger } = require('./middleware/logger')
const errorHandler = require('./middleware/errorHandler')
const cors = require('cors')
const corsOptions = require('./config/corsOptions')

const PORT = process.env.PORT || 3500

app.use(logger)
app.use(cors())
app.use(express.json())
app.use(fileUpload())

app.use('/', express.static(path.join(__dirname, '/public')))
// how index page is displayed 
app.use('/', require('./routes/root'))

app.use('/upload', require('./routes/uploadRoutes'))

// handles 404 pages 
app.all('*', (req, res) => {
    res.status(404)
    if (req.accepts('html')) {
        res.sendFile(path.join(__dirname, 'views', '404.html'))
    } else if (req.accepts('json')) {
        res.json({ message: '404 Not Found' })
    } else {
        res.type('txt').send('404 Not Found')
    }
})

app.use(errorHandler)
app.listen(PORT, () => console.log(`Server running on port ${PORT}`))