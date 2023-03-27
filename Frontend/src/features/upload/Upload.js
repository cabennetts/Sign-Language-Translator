import React from 'react'
import { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import { useSelector } from 'react-redux'
import { usePostVideoMutation } from './uploadApiSlice'



export default function Upload() {
  // Might need this for the usePostVideoMutation() but can't get to work

  const [postVideo, {
    isLoading,
    isSuccess,
    isError,
    error
  }] = usePostVideoMutation()

  const [selectedFile, setSelectedFile] = useState({});
  // const [selectedFile, setSelectedFile] = useState({});
  // const [isFilePicked, setIsFilePicked] = useState(false);
  
  const formData = new FormData()
  const changeHandler = (event) => {
    // console.log(event.target.files[0])
    setSelectedFile(event.target.files[0]);
    // setIsFilePicked(true);
  }

  const handleSubmission = async () => {
    let path
    const fileName = selectedFile.name
    formData.append('videoToUpload', selectedFile);
    // await postVideo({ formData })
    fetch('http://localhost:3500/upload', 
      {
        method: 'POST',
        body: formData,
      }
    ).then((response) => response.json())
     .then((result) => {
      path = result;
      console.log('Success (EXPRESS):', result);
     })
     .catch((error) => {
      console.error('Error (EXPRESS):', error);
     })

     // call python 
     fetch('http://localhost:8000/upload', 
      {
        method: 'GET',
        body: path,
        headers: {
          'Access-Control-Allow-Origin':'*'
        },
      }
    ).then((response) => response.json())
     .then((result) => {
      console.log('Success (FLASK):', result);
     })
     .catch((error) => {
      console.error('Error (FLASK):', error);
     })
  }


  return (
    <main>
        <h1>UPLOAD PAGE</h1>
        <p>Upload a video of you signing a phrase in ASL</p>
          
        <h3>Choose a video to interpret</h3>

        <input type="file" id='video' name="videoToUpload" onChange={changeHandler} className='btn-logo' />
        <button onClick={handleSubmission} type='submit' className='btn-green'>Upload</button>

        <h2>Results</h2>
        {/* <Interpretation /> */}
       

        <p id='results'></p>

    </main>
  )
}
