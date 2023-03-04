const express = require('express')
const router = express.Router()
const uploadController = require('../controllers/uploadController')
// routes to /upload 
router.route('/')
    .get(uploadController.getTest)
    .post(uploadController.postVideo)
    .patch(uploadController.updateInterp)

module.exports = router