include .env

.PHONY: run-react-code
run-react-code:
	@echo Starting react client
	cd frontend/network-dashboard && \
		npm run dev

.PHONY: run-backend
run-backend:
	@echo Starting backend server
	cd backend && \
	uvicorn server:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload

.PHONY: run-healing-server
run-healing-server:
	@echo Starting healing server
	cd 'Healing Service' && \
	uvicorn server:app --host 0.0.0.0 --port $(HEALING_SERVER_PORT) --reload

.PHONY: run-azure-healing-server
run-azure-healing-server:
	@echo Starting azure healing server
	cd 'Healing Service' && \
	uvicorn server2:app --host 0.0.0.0 --port $(AZURE_SERVER_PORT) --reload

.PHONY: run-system-health-server
run-system-health-server:
	@echo Starting system health server
	cd 'System Health Service' && \
	uvicorn server:app --host 0.0.0.0 --port $(SYSTEM_HEALTH_SERVER_PORT)  --reload


.PHONY: run-rag-server
run-rag-server:
	@echo Starting rag server
	cd 'Retriever Service' && \
	uvicorn server:app --host 0.0.0.0 --port $(RAG_PORT) --reload


.PHONY: run-azure-rag-server
run-azure-rag-server:
	@echo Starting azure rag healing server
	cd 'Healing Service' && \
	uvicorn server3:app --host 0.0.0.0 --port $(AZURE_SERVER_PORT) --reload