import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout';
import Public from './components/Public';
import Upload from './features/upload/Upload';
import Record from './features/record/Record';
import { Navbar } from './components/Navbar';
import './index.css'
function App() {

  // Entire application with components and routing
  return (
    <>
    <Navbar />
    <Routes>
      {/* Parent Route */}
      <Route path='/' element={<Layout />}>
        
        <Route index element={<Public />} />
        <Route path="upload" element={<Upload />} />
        <Route path="record" element={<Record />} />

      </Route>
    </Routes>
    </>
  )
}

export default App;
