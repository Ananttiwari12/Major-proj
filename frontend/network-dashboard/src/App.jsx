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

const featureNames = [
  "Seq",
  "Dur",
  "sHops",
  "dHops",
  "SrcPkts",
  "TotBytes",
  "SrcBytes",
  "Offset",
  "sMeanPktSz",
  "dMeanPktSz",
  "TcpRtt",
  "AckDat",
  "sTtl_",
  "dTtl_",
  "Proto_tcp",
  "Proto_udp",
  "Cause_Status",
  "State_INT",
];

function App() {
  const [chartData, setChartData] = useState([]);
  const [anomalyData, setAnomalyData] = useState(null);
  const [mitigation, setMitigation] = useState("");

  const introduceAnomaly = async () => {
    try {
      await fetch("http://localhost:8000/introduce_anomaly", {
        method: "POST",
      });
    } catch (error) {
      console.error("Error introducing anomaly:", error);
    }
  };

  const clearData = () => {
    setAnomalyData(null);
    setMitigation("");
  };

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/monitor");

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      // Update chart
      setChartData((prev) => [
        ...prev.slice(-49),
        {
          time: new Date(data.timestamp).toLocaleTimeString(),
          probability: data.probability,
        },
      ]);

      // Handle anomaly
      if (data.anomaly === 1) {
        setAnomalyData(data.features);
        try {
          const response = await fetch(
            `http://localhost:8080/heal?anomaly=${encodeURIComponent(
              JSON.stringify(data.features)
            )}`
          );
          setMitigation(await response.text());
        } catch (error) {
          setMitigation(`Error: ${error.message}`);
        }
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div className="app-container">
      <h1>Network Traffic Anomaly Detection</h1>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="probability"
              stroke="#ff7300"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="controls">
        <button onClick={introduceAnomaly}>Trigger Anomaly</button>
        <button onClick={clearData}>Clear</button>
      </div>

      {mitigation && (
        <div className="mitigation-panel">
          <h3>Recommended Mitigation</h3>
          <div className="mitigation-text">{mitigation}</div>
        </div>
      )}

      {anomalyData && (
        <div className="anomaly-panel">
          <h3>Anomaly Features</h3>
          <div className="features-grid">
            {featureNames.map((name) => (
              <div key={name} className="feature-item">
                <span className="feature-name">{name}</span>
                <span className="feature-value">
                  {typeof anomalyData[name] === "number"
                    ? anomalyData[name].toFixed(6)
                    : anomalyData[name]}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
