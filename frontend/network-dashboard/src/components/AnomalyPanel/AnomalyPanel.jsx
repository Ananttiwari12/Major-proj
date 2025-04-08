import React from "react";
import { featureNames } from "../../constants/featureNames";
import "./AnomalyPanel.css";

const AnomalyPanel = ({ anomalyData }) => {
  if (!anomalyData) return null;

  return (
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
  );
};

export default AnomalyPanel;
