import React from 'react'
import { Link } from "react-router-dom";
import {RiHome6Line} from 'react-icons/ri';
import {BsFillCloudUploadFill} from 'react-icons/bs';
import {BsFillRecordCircleFill} from 'react-icons/bs';

export const Navbar = () => {
  return (
    <nav className='navbar'>
        <ul>
            <li>
                <img src="/SLI_Project_Logo.png" alt="image" width='90' height='90'/>
                &nbsp;
                &nbsp;
            </li>
            <li>
                <button className="btn-logo">
                    <Link to="/">
                        Home
                        &nbsp;
                        <RiHome6Line  size={20}/>
                    </Link>
                </button>
            </li>

            <li>
                <button className="btn-blue">
                    <Link to="/upload">
                        Upload
                        &nbsp;
                        <BsFillCloudUploadFill size={20}/>
                    </Link>
                </button>
            </li>

            <li>
                <button className="btn-red">
                    <Link to="/record">
                        Record
                        &nbsp;
                        <BsFillRecordCircleFill size={20}/>
                    </Link>
                </button> 
            </li>
        </ul>
        
    </nav>
  )
}
