import { useState, useEffect } from "react";
import Chart from "./components/Chart/Chart";
import Controls from "./components/Controls/Controls";
import MitigationPanel from "./components/MitigationPanel/MitigationPanel";
import AnomalyPanel from "./components/AnomalyPanel/AnomalyPanel";
import "./App.css";

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

      <Chart chartData={chartData} />

      <Controls onIntroduceAnomaly={introduceAnomaly} onClearData={clearData} />

      <MitigationPanel mitigation={mitigation} />
      <AnomalyPanel anomalyData={anomalyData} />
    </div>
  );
}

export default App;
