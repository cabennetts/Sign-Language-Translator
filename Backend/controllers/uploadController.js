const express = require('express');
const path = require('path')
// for uploads
const fileUpload = require('express-fileupload')
// keep from using lots of try catch 
const asyncHandler = require('express-async-handler')

// @desc Upload a video
// @route POST /upload
const postVideo = asyncHandler(async (req, res) => {
    selectedFile = req.files.videoToUpload
    fileUpload({ createParentPath: true });
    
    let videoFile;
    let uploadPath;

    // check if file exists
    if (!req.files || Object.keys(req.files).length === 0) {
        return res.status(400).send('No files were uploaded.');
    }

    // store file in object
    videoFile = req.files.videoToUpload;

    // path to be uploaded to 
    uploadPath = path.join(__dirname, 'files', req.files.videoToUpload.name)
    
    // Use the mv() method to place the file somewhere on your server
    videoFile.mv(uploadPath, function(err) {
        if (err) { return res.status(500).send(err); }

        return res.status(200).json({ message: `File uploaded!, ${uploadPath}` });
    });
})

module.exports = {
    postVideo
}