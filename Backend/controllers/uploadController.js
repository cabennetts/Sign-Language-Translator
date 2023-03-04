const express = require('express');
const path = require('path')
const multer  = require('multer')
const upload = multer({ dest: '../files/' })

// const {spawn} = require('child_process')

// for uploads
const fileUpload = require('express-fileupload')
// keep from using lots of try catch 
const asyncHandler = require('express-async-handler')


// @desc get request for testing
// @route GET /upload
const getTest = asyncHandler(async (req, res) => {
    return res.status(200).json({ message: 'Connected' })
})

// @desc Upload a video
// @route POST /upload
const postVideo = asyncHandler(async (req, res) => {
    selectedFile  = req.files.videoToUpload
    fileUpload({ createParentPath: true });
    
    let videoFile;
    let uploadPath;

    // check if file exists
    if (!req.files || Object.keys(req.files).length === 0) {
        return res.status(400).send('No files were uploaded.');
    }

    // store file in object
    videoFile = req.files.videoToUpload;
    // console.log('videoFile', videoFile)

    // path to be uploaded to 
    // uploadPath = __dirname + '/files/' + videoFile.name;
    uploadPath = path.join(__dirname, 'files', req.files.videoToUpload.name)
    
    // Use the mv() method to place the file somewhere on your server
    videoFile.mv(uploadPath, function(err) {
        if (err) { return res.status(500).send(err); }

        // var dataToSend;
        // // spawn new child process to call the python script
        // const python = spawn('python', ['hw.py'])
        // // spawn('python', ['hw.py'])
        // // is equivalent to 'python3 hw.py'
        
        // // collect data from script
        // python.stdout.on('data', (data) => {
        //     console.log('Pipe data from python script ...');
        //     dataToSend = data.toString();
        // });

        // // in close event we are sure that stream from child process is closed
        // python.on('close', (code) => {
        //     console.log(`child process close all stdio with code ${code}`)
        //     // res.send(dataToSend)
        // })

        // Call python script here that will process the video
        return res.status(200).json({ message: `File uploaded!, ${dataToSend}` });
    });
})

// @desc Update interpretation
// @route PATCH /upload
const updateInterp = asyncHandler(async (req, res) => {

})

module.exports = {
    getTest,
    postVideo,
    updateInterp
}