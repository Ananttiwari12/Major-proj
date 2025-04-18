import React from "react";
import "./MitigationPanel.css";

const MitigationPanel = ({ mitigation }) => {
  if (!mitigation) return null;

  // Process the mitigation string to extract structured sections
  const parseMitigation = (mitigationString) => {
    try {
      // Clean JSON formatting if present
      let cleanedString = mitigationString;
      if (mitigationString.startsWith("[") && mitigationString.endsWith("]")) {
        cleanedString = mitigationString.substring(
          1,
          mitigationString.length - 1
        );
        try {
          const parsed = JSON.parse("[" + cleanedString + "]");
          if (Array.isArray(parsed) && parsed.length > 0) {
            cleanedString = parsed[0];
          }
        } catch (e) {
          console.log("JSON parsing failed, using raw string");
        }
      }

      // Replace escaped newlines with actual line breaks
      cleanedString = cleanedString.replace(/\\n/g, "\n");

      // Extract sections based on numbered format
      const sections = {
        threatAssessment: "",
        mitigationCommand: "",
        parameters: {},
        verification: "",
        fallback: "",
      };

      // Regular expressions to match each section
      const threatRegex = /1\.\s*Threat Assessment:\s*(.*?)(?=2\.|$)/s;
      const mitigationRegex = /2\.\s*Mitigation Command:\s*(.*?)(?=3\.|$)/s;
      const parametersRegex = /3\.\s*Parameters:\s*(.*?)(?=4\.|$)/s;
      const verificationRegex = /4\.\s*Verification:\s*(.*?)(?=5\.|$)/s;
      const fallbackRegex = /5\.\s*Fallback:\s*(.*?)(?=$)/s;

      // Extract each section
      const threatMatch = cleanedString.match(threatRegex);
      const mitigationMatch = cleanedString.match(mitigationRegex);
      const parametersMatch = cleanedString.match(parametersRegex);
      const verificationMatch = cleanedString.match(verificationRegex);
      const fallbackMatch = cleanedString.match(fallbackRegex);

      if (threatMatch) sections.threatAssessment = threatMatch[1].trim();
      if (mitigationMatch)
        sections.mitigationCommand = mitigationMatch[1].trim();

      // Extract parameters
      if (parametersMatch) {
        const paramText = parametersMatch[1];
        const targetMatch = paramText.match(/a\.\s*Target:\s*(.*?)(?=b\.|$)/s);
        const actionMatch = paramText.match(/b\.\s*Action:\s*(.*?)(?=c\.|$)/s);
        const durationMatch = paramText.match(
          /c\.\s*Duration:\s*(.*?)(?=d\.|$)/s
        );
        const severityMatch = paramText.match(/d\.\s*Severity:\s*(.*?)(?=$)/s);

        if (targetMatch) sections.parameters.target = targetMatch[1].trim();
        if (actionMatch) sections.parameters.action = actionMatch[1].trim();
        if (durationMatch)
          sections.parameters.duration = durationMatch[1].trim();
        if (severityMatch)
          sections.parameters.severity = severityMatch[1].trim();
      }

      if (verificationMatch)
        sections.verification = verificationMatch[1].trim();
      if (fallbackMatch) sections.fallback = fallbackMatch[1].trim();

      return sections;
    } catch (error) {
      console.error("Error parsing mitigation:", error);
      return { raw: mitigationString };
    }
  };

  const sections = parseMitigation(mitigation);

  // Get severity class based on severity level
  const getSeverityClass = (severity) => {
    if (!severity) return "";
    const severityLower = severity.toLowerCase();
    if (severityLower.includes("critical")) return "severity-critical";
    if (severityLower.includes("high")) return "severity-high";
    if (severityLower.includes("medium")) return "severity-medium";
    if (severityLower.includes("low")) return "severity-low";
    return "";
  };

  // Display raw text if parsing failed
  if (sections.raw) {
    return (
      <div className="mitigation-panel">
        <h3>Recommended Mitigation</h3>
        <div className="mitigation-text">{sections.raw}</div>
      </div>
    );
  }

  return (
    <div className="mitigation-panel">
      <h3>Automated Mitigation Response</h3>

      {sections.threatAssessment && (
        <div className="mitigation-section">
          <h4>Threat Assessment</h4>
          <p>{sections.threatAssessment}</p>
        </div>
      )}

      {sections.mitigationCommand && (
        <div className="mitigation-section">
          <h4>Mitigation Command</h4>
          <div className="command-box">{sections.mitigationCommand}</div>
        </div>
      )}

      {sections.parameters && Object.keys(sections.parameters).length > 0 && (
        <div className="mitigation-section">
          <h4>Parameters</h4>
          <div className="parameters-grid">
            {sections.parameters.target && (
              <div>
                <strong>Target:</strong> {sections.parameters.target}
              </div>
            )}
            {sections.parameters.action && (
              <div>
                <strong>Action:</strong> {sections.parameters.action}
              </div>
            )}
            {sections.parameters.duration && (
              <div>
                <strong>Duration:</strong> {sections.parameters.duration}
              </div>
            )}
            {sections.parameters.severity && (
              <div
                className={`severity ${getSeverityClass(
                  sections.parameters.severity
                )}`}
              >
                <strong>Severity:</strong> {sections.parameters.severity}
              </div>
            )}
          </div>
        </div>
      )}

      {sections.verification && (
        <div className="mitigation-section">
          <h4>Verification</h4>
          <p>{sections.verification}</p>
        </div>
      )}

      {sections.fallback && (
        <div className="mitigation-section">
          <h4>Fallback</h4>
          <p>{sections.fallback}</p>
        </div>
      )}

      <div className="execution-status">
        <div className="status-indicator"></div>
        <span>Ready for execution</span>
      </div>
    </div>
  );
};

export default MitigationPanel;
