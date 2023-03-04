import React from 'react'
import { Link } from "react-router-dom";

export const Navbar = () => {
  return (
    <nav className='navbar'>
        <ul>
            <li>
                <button className="btn-logo">
                    <Link to="/">Home</Link>
                </button>
            </li>

            <li>
                <button className="btn-red">
                    <Link to="/upload">Upload</Link>
                </button>
            </li>

            <li>
                <button className="btn-blue">
                    <Link to="/record">Record</Link>
                </button> 
            </li>
        </ul>
        
    </nav>
  )
}
