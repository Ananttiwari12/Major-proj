import { useState, useEffect } from "react";
import Chart from "./components/Chart/Chart";
import Controls from "./components/Controls/Controls";
import MitigationPanel from "./components/MitigationPanel/MitigationPanel";
import AnomalyPanel from "./components/AnomalyPanel/AnomalyPanel";
import SystemHealth from "./components/SystemHealth/SystemHealth";
import "./App.css";

function App() {
  const [chartData, setChartData] = useState([]);
  const [anomalyData, setAnomalyData] = useState(null);
  const [mitigation, setMitigation] = useState("");
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 0,
    memory: 0,
    upload: 0,
    download: 0,
    services: {
      detector: "checking",
      mitigator: "checking",
      database: "checking",
    },
  });

  const introduceAnomaly = async () => {
    try {
      await fetch(
        `http://localhost:${
          import.meta.env.VITE_BACKEND_SERVER_PORT
        }/introduce_anomaly`,
        {
          method: "POST",
        }
      );
    } catch (error) {
      console.error("Error introducing anomaly:", error);
    }
  };

  const clearData = () => {
    setAnomalyData(null);
    setMitigation("");
  };

  // WebSocket for anomaly detection
  useEffect(() => {
    const ws = new WebSocket(
      `ws://localhost:${import.meta.env.VITE_BACKEND_SERVER_PORT}/ws/monitor`
    );

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
            `http://localhost:${
              import.meta.env.VITE_LLM_PORT
            }/heal?anomaly=${encodeURIComponent(JSON.stringify(data.features))}`
          );
          setMitigation(await response.text());
        } catch (error) {
          setMitigation(`Error: ${error.message}`);
        }
      }
    };

    return () => ws.close();
  }, []);

  // System health monitoring
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch(
          `http://localhost:${
            import.meta.env.VITE_SYSTEM_HEALTH_SERVICE_PORT
          }/system_metrics`
        );
        setSystemMetrics(await res.json());
      } catch (error) {
        console.error("Failed to fetch metrics:", error);
        setSystemMetrics((prev) => ({
          ...prev,
          services: {
            detector: "down",
            mitigator: "down",
            database: "down",
          },
        }));
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <h1>Network Traffic Anomaly Detection</h1>

      <div className="dashboard-grid">
        <div className="main-content">
          <Chart chartData={chartData} />
          <Controls
            onIntroduceAnomaly={introduceAnomaly}
            onClearData={clearData}
          />
          {mitigation && <MitigationPanel mitigation={mitigation} />}
          {anomalyData && <AnomalyPanel anomalyData={anomalyData} />}
        </div>

        <div className="sidebar">
          <SystemHealth metrics={systemMetrics} />
        </div>
      </div>
    </div>
  );
}

export default App;
