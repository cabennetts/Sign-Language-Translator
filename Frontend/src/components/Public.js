import React from 'react'
import '../index.css'

const Public = () => {
    const content = (
        <>
          
            <main>
                <h1> Sign Language Interpreter </h1>
                <p>Here you can upload a video or record one of you performing a phrase or sentence in ASL.
                    The video will run through some data processing phases and then tested against our trained 
                    model to give you the english translation of what was signed!
                </p>          
            </main>
        </>
    )

    return content
}

export default Public