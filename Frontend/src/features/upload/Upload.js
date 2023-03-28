import React from 'react'
import UploadVideo from '../../components/UploadVideo'
import InterpretVideo from '../../components/InterpretVideo'

export default function Upload() {
 
  return (
    <>
      <main>
        <h1>UPLOAD PAGE</h1>

        <p>Upload a video of you signing a phrase in ASL</p>
        <UploadVideo />

        <p>Interpret your video</p>
        <InterpretVideo />
       
      </main>
    </>
  )
}
