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
    formData.append('videoToUpload', selectedFile);

    // await postVideo({ formData })
    fetch('http://localhost:3500/upload', 
      {
        method: 'POST',
        body: formData,
      }
    ).then((response) => response.json())
     .then((result) => {
      console.log('Success:', result);
     })
     .catch((error) => {
      console.error('Error:', error);
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
        <h4> </h4>
        <p id='results'></p>

    </main>
  )
}



/////////////////////////////////////
// import React from 'react'
// import { useState, useEffect } from "react"
// import { usePostVideoMutation } from './uploadApiSlice'
// import { useNavigate } from 'react-router-dom'
// import { useSelector } from 'react-redux'


// const UploadForm = () => {
//   const [addNewVideo, {
//     isLoading,
//     isSuccess,
//     isError, 
//     error
//   }] = usePostVideoMutation()

//   const navigate = useNavigate()

//   const [videofile, setVideofile] = useState('')
  
//   useEffect(() => {
//     if (isSuccess) {
//         setVideofile('')
//         navigate('/upload')
//     }
//   }, [isSuccess, navigate])

//   const onUploadClicked = async (e) => {
//     e.preventDefault()
//     await addNewVideo({ videofile })
//   }

// }


// export default function Upload() {
 
//   // // get form
//   // const form  = document.getElementById('uploadForm')
//   // // handle file upload 
//   // const sendFile = async () => {
//   //   // object
//   //   const videoFile = document.getElementById('video')
//   //   const formData = new FormData()
//   //   Object.keys(videoFile).forEach(key => {
//   //     formData.append(videoFile)
//   //   })
//   //   const response = await fetch('http://localhost:3500/upload', {
//   //     method: 'GET',
//   //     body: formData
//   //   })
//   //   const json = await response.json()
//   //   // Request status (accept, fail, error)
//   //   const h4 = document.querySelector('h4')
//   //   h4.textContent = `Status: ${json?.status}`
//   //   // // results from running the video through the model
//   //   // const p = document.getElementById('results')
//   //   // p.textContent = `We interpreted: ${json?.message}`
//   // }


//   // if (form) {
//   //   form.addEventListener('submit', (e) => {
//   //     e.preventDefault()
//   //     sendFile()
//   //   })
//   // }
//   const [addNewVideo, {
//     isLoading,
//     isSuccess,
//     isError, 
//     error
//   }] = usePostVideoMutation()

//   const navigate = useNavigate()

//   const [videofile, setVideofile] = useState('')
  
//   useEffect(() => {
//     if (isSuccess) {
//         setVideofile('')
//         navigate('/upload')
//     }
//   }, [isSuccess, navigate])

//   const onUploadClicked = async (e) => {
//     e.preventDefault()
//     await addNewVideo({ videofile })
//   }

//   const content = (
//     <>
    
//     <main>
//         <h1>UPLOAD PAGE</h1>
//         <p>Upload a video of you signing a phrase in ASL</p>
//         {/* <UploadForm /> */}
//         <form id="uploadForm" onSubmit={onUploadClicked}>
//           <label>
//             <h3>Choose a video to interpret</h3>
//             <input type="file" name="videoFile" id='video'  accept='video/mp4' className='btn-logo'></input>
//           </label>
//           <button type='submit' value="Submit" className='btn-green'>Submit</button>

//           {/* <button type="submit" value="Submit" className='btn-green'>Upload</button> */}
//         </form>
    
        
//         <h2>Results</h2>
//         {/* <Interpretation /> */}
//         <h4> </h4>
//         <p id='results'></p>

//     </main>
//     </>
//   )
//   return content
// }
