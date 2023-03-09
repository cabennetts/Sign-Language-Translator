import React from 'react'
import '../index.css'

const Public = () => {
    const content = (
        <>
          
            <main>
                <h1> Sign Language Interpreter </h1>
                <p></p>
                <img src="/SLI_Project_Logo.png" alt="image" width='300' height='300'/>
                <h1>About</h1>
                <p>
                    For our senior design project at The University of Kansas, we decided to build a sign 
                    language interpreter using computer vision and machine learning. Our goal was to create
                    an online application that recognizes sign language signs and interprets them to English text
                    to facilitate easier and/or more efficient communication with individuals with hearing impairments.
                </p>
                <p>
                    Here you can upload a video or record one of you performing a phrase or sentence in ASL.
                    The video will run through some data processing phases and then tested against our trained 
                    model to give you the english translation of what was signed!
                    <b>** add more technical information about models/routing/interpretation process**</b>
                </p>          
                <h1>Team</h1>
                <table>
                    <tr>
                        <td><img src="/SLI_Project_Logo.png" alt="image" width='300' height='300'/></td>
                        <td><img src="/SLI_Project_Logo.png" alt="image" width='300' height='300'/></td>
                        <td><img src="/SLI_Project_Logo.png" alt="image" width='300' height='300'/></td>
                        <td><img src="/SLI_Project_Logo.png" alt="image" width='300' height='300'/></td>
                        <td><img src="/SLI_Project_Logo.png" alt="image" width='300' height='300'/></td>
                    </tr>
                    <tr>
                        <th>Alex Anderson</th>
                        <th>Ben Lottes</th>
                        <th>Bolu Adubi</th>
                        <th>Caleb Bennetts</th>
                        <th>Matrim Besch</th>
                    </tr>
                </table>
                <h1>GitHub Repository</h1>
                <p>
                    All our work can be found at this link:
                </p>
            </main>
        </>
    )

    return content
}

export default Public