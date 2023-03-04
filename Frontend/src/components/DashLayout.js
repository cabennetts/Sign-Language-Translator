import { Outlet } from "react-router-dom"
import DashHeader from "./DashHeader"

// Layout component for protected parts of the site
const DashLayout = () => {
  return (
    <>
        {/*  */}
        <DashHeader />

        <div>
            <Outlet />
        </div>
    </>
  )
}

export default DashLayout