const express = require('express')
const router = express.Router()
const uploadController = require('../controllers/uploadController')
// routes to /upload 
router.route('/')
    .post(uploadController.postVideo)

module.exports = router