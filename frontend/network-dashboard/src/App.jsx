import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

function App() {
  const [data, setData] = useState([]);
  const [mitigation, setMitigation] = useState(""); // Store mitigation response

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/monitor");

    ws.onopen = () => console.log("Connected to WebSocket");

    ws.onmessage = async (event) => {
      const newData = JSON.parse(event.data);
      setData((prevData) =>
        [
          ...prevData,
          {
            time: new Date(newData.timestamp).toLocaleTimeString(),
            probability: newData.probability,
          },
        ].slice(-50)
      );

      // If anomaly detected, display mitigation strategy
      if (newData.anomaly === 1) {
        setMitigation(newData["mitigation"]); // Update state with LLM response
      }
    };

    ws.onerror = (error) => console.error("WebSocket error:", error);
    ws.onclose = () => console.log("WebSocket connection closed");

    return () => ws.close();
  }, []);

  const introduceAnomaly = async () => {
    try {
      const response = await fetch("http://localhost:8000/introduce_anomaly", {
        method: "POST",
      });
      const result = await response.json();
      console.log("Anomaly introduced:", result);
    } catch (error) {
      console.error("Error introducing anomaly:", error);
    }
  };

  return (
    <div className="container">
      <h1>Network Monitoring Dashboard</h1>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" tick={{ fontSize: 12 }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="probability"
            stroke="#ff7300"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>

      <button onClick={introduceAnomaly}>Introduce Anomaly</button>

      {mitigation && (
        <div className="mitigation-box">
          <h3>Mitigation Response:</h3>
          <p>{mitigation}</p>
        </div>
      )}
    </div>
  );
}

export default App;
