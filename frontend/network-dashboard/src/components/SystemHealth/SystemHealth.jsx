import React from "react";
import { RadialBarChart, RadialBar, ResponsiveContainer } from "recharts";
import "./SystemHealth.css";

const SystemHealth = ({ metrics }) => {
  const cpuData = [{ value: metrics.cpu, fill: "#8884d8" }];
  const memoryData = [{ value: metrics.memory, fill: "#82ca9d" }];
  const networkData = [{ value: metrics.network, fill: "#ffc658" }];

  return (
    <div className="health-panel">
      <h3>System Health</h3>
      <div className="health-metrics">
        <div className="metric">
          <h4>CPU Usage</h4>
          <ResponsiveContainer width={100} height={100}>
            <RadialBarChart
              innerRadius={20}
              outerRadius={80}
              data={cpuData}
              startAngle={90}
              endAngle={-270}
            >
              <RadialBar background dataKey="value" />
              <text x={50} y={50} textAnchor="middle" dominantBaseline="middle">
                {metrics.cpu}%
              </text>
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        <div className="metric">
          <h4>Memory</h4>
          <ResponsiveContainer width={100} height={100}>
            <RadialBarChart
              innerRadius={20}
              outerRadius={80}
              data={cpuData}
              startAngle={90}
              endAngle={-270}
            >
              <RadialBar background dataKey="value" />
              <text x={50} y={50} textAnchor="middle" dominantBaseline="middle">
                {metrics.memory}%
              </text>
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        <div className="metric">
          <h4>Network</h4>
          <div className="network-stats">
            <span>↑ {metrics.upload} Mbps</span>
            <span>↓ {metrics.download} Mbps</span>
          </div>
        </div>

        <div className="service-status">
          <h4>Services</h4>
          <ul>
            <li className={metrics.services.detector}>
              Detector {metrics.services.detector === "up" ? "✓" : "✗"}
            </li>
            <li className={metrics.services.mitigator}>
              Mitigator {metrics.services.mitigator === "up" ? "✓" : "✗"}
            </li>
            <li className={metrics.services.database}>
              Database {metrics.services.database === "up" ? "✓" : "✗"}
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SystemHealth;
