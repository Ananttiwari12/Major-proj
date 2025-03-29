.PHONY: run-react-code
run-react-code:
	@echo Starting react client
	cd frontend/network-dashboard && \
		npm run dev

.PHONY: run-backend
run-backend:
	@echo Starting backend server
	cd backend && \
	uvicorn server:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run-healing-server
run-healing-server:
	@echo Starting healing server
	cd 'Healing Service' && \
	uvicorn server:app --host 0.0.0.0 --port 8080 --reload

.PHONY: run-azure-healing-server
run-azure-healing-server:
	@echo Starting azure healing server
	cd 'Healing Service' && \
	uvicorn server2:app --host 0.0.0.0 --port 8080 --reload