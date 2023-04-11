import React from 'react'
import { useState } from 'react'

// Handles actually processing video uploaded
const InterpretVideo = () => {
    
    // hook for getting result
    const [interp, setInterp] = useState([])
    const [words, setWords] = useState({})
    const [english, setEnglish] = useState([])

    const handleInterpretation = () => {
        // call python 
        fetch('http://localhost:8000/upload', 
        {
          method: 'GET',
          headers: {
            'Access-Control-Allow-Origin':'*'
          },
        }
        ).then((response) => {
          if(response.ok){
            return response.json()
          }
          throw response;
        })
        .then((data) => {
          console.log('Success (FLASK):', data);
          setInterp(data.result)
          setWords(data.words)
          setEnglish(data.english)
        })
        .catch((error) => {
          console.error('Error (FLASK):', error);
        })
      }
      
    return (
        <>
            <button onClick={handleInterpretation} type='submit' className='btn-blue'>Interpret</button> 
            <h5>Interpreted String:</h5>
            {interp.length > 0 && (
                <p> {interp.toString()} </p>
            )}
            
            <br></br>

            <h5>English Translation:</h5>
            {english.length > 0 && (
              <p> {english.toString()} </p>
            )}
            <br></br>
            
            <table className='resTable'>
                <tr>
                    <th>Word</th>
                    <th>Average Accuracy</th>
                </tr>
                {Object.keys(words).map((innerAttr, index) => {
                    return (
                        <tr key={index}>
                            <td> {innerAttr} </td>
                            <td> {words[innerAttr]} </td>
                        </tr>
                    )
                })}
            </table>
        </>
  )
}

export default InterpretVideo