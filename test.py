from src.RiskEngineWrapper import RiskEngineWrapper

if __name__=="__main__":

    rew = RiskEngineWrapper()
    rew.init_risk_engine("file.csv", "^990100-USD-STRD")