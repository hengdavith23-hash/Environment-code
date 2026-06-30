/*
 * ============================================================
 *  DQN-Based Resource Allocations for eMBB — NS-3 Environment
 * ============================================================
 */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/opengym-module.h"

#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("EmbbDqnEnv");

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
static const uint32_t NUM_UE        = 6;
static const uint32_t STATE_DIM     = 12;  
static const uint32_t ACTION_DIM    = 3;  

static double g_maxSysThroughput = NUM_UE * 10.24;
static double g_maxSinrDb        = 33.9; 
static double g_maxQueueBytes    = 1e-9; 
static double g_maxDelayMs       = 100.0;  

static const uint32_t STEPS_PER_EP  = 10;
static const uint32_t MAX_EPISODES  = 300;
static const uint32_t MAX_STEPS     = STEPS_PER_EP * MAX_EPISODES;

static const double   EP_TIME_S     = 10.0;
static const double   STEP_TIME_S   = EP_TIME_S / STEPS_PER_EP;  // 1.0s
static const double   WARMUP_S      = 0.5;
static const double   SIM_TIME_S    = MAX_EPISODES * EP_TIME_S + WARMUP_S ;

// ─────────────────────────────────────────────────────────────────────────────
static const double ACTION_MU[3]  = {1.0, 2.0, 1.0};
static const double ACTION_PHI[3] = {1.0, 1.0, 0.0};

// Current µ and φ — updated by ExecuteActions each step
static double g_mu  = 1.0;
static double g_phi = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
//  Per-UE runtime state
// ─────────────────────────────────────────────────────────────────────────────
struct UeStats {
    double   sinrDb         = 0.0;  // latest SINR (dB) from trace
    double   queueBytes     = 0.0;  // current step queue (bytes)

    double   throughputMbps = 0.0;  // step throughput
    double   delayMs        = 0.0;  // step delay

    // Episode accumulators
    double   epThroughput   = 0.0;
    double   epDelay        = 0.0;
    double   epAvgSinr      = 0.0;  // average SINR over episode steps
    double   epAvgQueue     = 0.0;  // average queue (bytes) over episode steps
    uint32_t epStepCount    = 0;    // steps counted in episode

    uint64_t epTxPkts       = 0;
    uint64_t epRxPkts       = 0;

    double   epRxBytesStart  = 0.0;
    double   epDelaySumStart = 0.0;
    uint64_t epTxPktsStart   = 0;
    uint64_t epRxPktsStart   = 0;

    double   rxBytesPrev    = 0.0;
    double   delaySumPrev   = 0.0;
    uint64_t txPktsPrev     = 0;
    uint64_t rxPktsPrev     = 0;
};

static std::vector<UeStats> g_ueStats(NUM_UE);

// ─────────────────────────────────────────────────────────────────────────────
//  Global handles
// ─────────────────────────────────────────────────────────────────────────────
static Ptr<FlowMonitor>        g_monitor;
static Ptr<Ipv4FlowClassifier> g_classifier;
static Ptr<OpenGymInterface>   g_openGym;
static uint32_t                g_currentAction = 0; 

static uint32_t g_stepCounter  = 0;
static uint32_t g_totalSteps   = 0;
static uint32_t g_episodeCount = 0;

// UE net devices
static NetDeviceContainer g_ueNetDev;

// UE mobility
static NodeContainer              g_ueNodes;
static Vector                     g_gnbPos;
static Ptr<UniformRandomVariable> g_randAngle  = nullptr;
static Ptr<UniformRandomVariable> g_randRadius = nullptr;

// Output
static std::string g_outputDir = "./";
static std::string g_simTag    = "default";

// ─────────────────────────────────────────────────────────────────────────────
//  SINR trace
// ─────────────────────────────────────────────────────────────────────────────
void SinrTrace(uint16_t /*cellId*/, uint16_t rnti, double avgSinr,
               uint16_t /*bwpId*/, uint8_t /*streamId*/)
{
    if (rnti > 0 && rnti <= NUM_UE)
    {
        double sinrDb = 10.0 * std::log10(avgSinr + 1e-12);
        sinrDb = std::max(-10.0, std::min(sinrDb, 33.9));
        g_ueStats[rnti - 1].sinrDb = sinrDb;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Jain Fairness Index
// ─────────────────────────────────────────────────────────────────────────────
double JainFairness(const std::vector<double>& x)
{
    double sum = 0, sumSq = 0;
    for (double v : x) { sum += v; sumSq += v * v; }
    if (sumSq < 1e-12) return 1.0;
    return (sum * sum) / ((double)x.size() * sumSq);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Apply DQN action — compute RB metric, store µ/φ
//  M_i = normDelay^µ × normSinr^φ
// ─────────────────────────────────────────────────────────────────────────────
void ApplyAction(uint32_t action)
{
    if (action >= ACTION_DIM) action = 0;
    g_mu  = ACTION_MU[action];
    g_phi = ACTION_PHI[action];

    static const char* names[] = {"(µ=1,φ=1)", "(µ=2,φ=1)", "(µ=1,φ=0)"};

    for (uint32_t i = 0; i < NUM_UE; ++i)
    {
        double normDelay = std::max(0.01,
            std::min(g_ueStats[i].delayMs / std::max(g_maxDelayMs, 1.0), 1.0));
        double normSinr  = std::max(0.01,
            std::min((g_ueStats[i].sinrDb + 10.0) / (g_maxSinrDb + 10.0), 1.0));
        double M = std::pow(normDelay, g_mu) * std::pow(normSinr, g_phi);
        NS_LOG_INFO("Action=" << action << " " << names[action]
            << " UE" << (i+1) << " delay=" << g_ueStats[i].delayMs //Queue apply base on delay
            << "ms sinr=" << g_ueStats[i].sinrDb << "dB M=" << M);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Write per-episode flow stats to file
// ─────────────────────────────────────────────────────────────────────────────
void WriteEpisodeStats(uint32_t epNum)
{
    std::string filename = g_outputDir + "/ep_" + std::to_string(epNum) + "_" + g_simTag + ".txt";
    std::ofstream f(filename, std::ofstream::out | std::ofstream::trunc);
    if (!f.is_open()) return;
    f << std::fixed << std::setprecision(6);
    f << "===== Episode " << epNum << "/" << MAX_EPISODES << " =====\n\n";

    double avgTput = 0, avgDelay = 0, avgPlr = 0;
    for (uint32_t i = 0; i < NUM_UE; ++i)
    {
        auto& u  = g_ueStats[i];
        double plr = (u.epTxPkts > 0 && u.epTxPkts >= u.epRxPkts)
                     ? (double)(u.epTxPkts - u.epRxPkts) * 100.0 / u.epTxPkts : 0.0;
        plr = std::max(0.0, std::min(plr, 100.0));

        f << "UE" << (i+1) << "\n";
        f << " Throughput:       " << u.epThroughput << " Mbps\n";
        f << " Mean delay:       " << u.epDelay      << " ms\n";
        f << " Packet Loss Rate: " << plr             << " %\n";
        f << " Mean Queue:       " << u.epAvgQueue    << " bytes\n\n";

        avgTput  += u.epThroughput;
        avgDelay += u.epDelay;
        avgPlr   += plr;
    }
    f << "Mean flow throughput:     " << avgTput  / NUM_UE << " Mbps\n";
    f << "Mean flow delay:          " << avgDelay / NUM_UE << " ms\n";
    f << "Average Packet Loss Rate: " << avgPlr   / NUM_UE << " %\n";
    f.close();

    // ── Append one row to summary CSV ─────────────────────────────────────────
    std::string csvFile = g_outputDir + "/results_summary02" + g_simTag + ".csv";
    bool newFile = !std::ifstream(csvFile).good();
    std::ofstream csv(csvFile, std::ofstream::app);
    if (csv.is_open())
    {
        if (newFile)
            csv << "Episode";
            for (uint32_t i = 0; i < NUM_UE; ++i) csv << ",UE" << (i+1) << "_Tput";
            csv << ",Mean_Tput";
            for (uint32_t i = 0; i < NUM_UE; ++i) csv << ",UE" << (i+1) << "_Delay";
            csv << ",Mean_Delay";
            for (uint32_t i = 0; i < NUM_UE; ++i) csv << ",UE" << (i+1) << "_PLR";
            csv << ",Mean_PLR";
            for (uint32_t i = 0; i < NUM_UE; ++i) csv << ",UE" << (i+1) << "_SINR";
            csv << "\n";

        csv << std::fixed << std::setprecision(4);
        csv << epNum;
        for (uint32_t i = 0; i < NUM_UE; ++i)
            csv << "," << g_ueStats[i].epThroughput;
        csv << "," << avgTput / NUM_UE;
        for (uint32_t i = 0; i < NUM_UE; ++i)
            csv << "," << g_ueStats[i].epDelay;
        csv << "," << avgDelay / NUM_UE;
        for (uint32_t i = 0; i < NUM_UE; ++i) {
            double plr = (g_ueStats[i].epTxPkts > 0 && g_ueStats[i].epTxPkts >= g_ueStats[i].epRxPkts)
                         ? (double)(g_ueStats[i].epTxPkts - g_ueStats[i].epRxPkts) * 100.0 / g_ueStats[i].epTxPkts : 0.0;
            csv << "," << std::max(0.0, std::min(plr, 100.0));
        }
        csv << "," << avgPlr / NUM_UE;
        for (uint32_t i = 0; i < NUM_UE; ++i)
            csv << "," << g_ueStats[i].sinrDb;
        csv << "\n";
        csv.close();
    }

    // Console print
    std::cout << "\n===== Episode " << epNum << "/" << MAX_EPISODES << " Results =====\n";
    for (uint32_t i = 0; i < NUM_UE; ++i)
    {
        auto& u  = g_ueStats[i];
        double plr = (u.epTxPkts > 0 && u.epTxPkts >= u.epRxPkts)
                     ? (double)(u.epTxPkts - u.epRxPkts) * 100.0 / u.epTxPkts : 0.0;
        plr = std::max(0.0, std::min(plr, 100.0));
        std::cout << "  UE" << (i+1)
                  << " | Tput="  << std::setprecision(3) << u.epThroughput  << " Mbps"
                  << " | Delay=" << std::setprecision(3) << u.epDelay       << " ms"
                  << " | PLR="   << std::setprecision(3) << plr             << " %\n";
    }
    std::cout << "  Mean Tput=" << avgTput/NUM_UE << " Mbps"
              << " | Mean Delay=" << avgDelay/NUM_UE << " ms"
              << " | Mean PLR=" << avgPlr/NUM_UE << " %\n";
    std::cout << "===========================================\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  OpenGym callbacks
// ─────────────────────────────────────────────────────────────────────────────
Ptr<OpenGymSpace> GetObservationSpace()
{
    std::vector<uint32_t> shape = {STATE_DIM};
    return CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, shape, "float32");
}

Ptr<OpenGymSpace> GetActionSpace()
{
    return CreateObject<OpenGymDiscreteSpace>(ACTION_DIM);
}

// State = [SINR_norm×4, Queue_norm×4]
Ptr<OpenGymDataContainer> GetObservation()
{
    std::vector<uint32_t> shape = {STATE_DIM};
    Ptr<OpenGymBoxContainer<float>> box =
        CreateObject<OpenGymBoxContainer<float>>(shape);

    // SINR normalised [0,1] — dynamic max
    for (uint32_t i = 0; i < NUM_UE; ++i)
        box->AddValue((float)std::max(0.0,
            std::min(g_ueStats[i].sinrDb / g_maxSinrDb, 1.0)));

    // Queue normalised [0,1] — dynamic max
    for (uint32_t i = 0; i < NUM_UE; ++i)
        box->AddValue((float)std::min(g_ueStats[i].queueBytes / g_maxQueueBytes, 1.0));

    return box;
}

// Reward = NorSysThroughput + Fairness − NormDelay
// ∑R_i(t) = Normalized System Throughput, 0≤∑R_i(t)≤1
// ∑q_i(t) = Normalized System Throughput, 0≤∑q_i(t)≤1
//F = Jain's Fairness Index, 0≤F≤1.// λ = Delay penalty coefficient (typically λ=1)
// Rt​=∑​R_i​(t)+F−λ∑q_i​(t)
float GetReward()
{
    double sysTh = 0.0, avgDelay = 0.0;
    std::vector<double> thVec;
    for (auto& u : g_ueStats) {
        sysTh    += u.throughputMbps;
        avgDelay += u.delayMs;
        thVec.push_back(u.throughputMbps);
    }
    double norSysTh  = std::min(sysTh / g_maxSysThroughput, 1.0);
    double fairness  = JainFairness(thVec);
    double normDelay = std::min((avgDelay / NUM_UE) / g_maxDelayMs, 1.0);
    return (float)(norSysTh + fairness - normDelay);
}

std::string GetExtraInfo() { return ""; }

// Forward declaration
void StepCallback();

// GetGameOver: true only after all steps across all episodes are done.
// g_totalSteps is incremented AFTER NotifyCurrentState() in StepCallback,
// so this function sees the pre-increment value during step N's ZMQ round-trip.
// On step MAX_STEPS, g_totalSteps becomes MAX_STEPS only after notify returns —
// meaning the agent receives gameOver=true at the start of step MAX_STEPS+1
// (which ExecuteActions schedules but simTime prevents from running).
// Net effect: all MAX_STEPS steps execute and all episodes are written fully.
bool GetGameOver()
{
    return (g_totalSteps >= MAX_STEPS);
}

// ExecuteActions: apply DQN decision, always schedule next step.
// Never call Simulator::Stop() here — the simulation ends naturally at
// simTime. ScheduleNextStateRead-style self-scheduling ensures every step
// across every episode runs to completion.
bool ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> disc =
        DynamicCast<OpenGymDiscreteContainer>(action);
    g_currentAction = disc->GetValue();
    ApplyAction(g_currentAction);

    // Always schedule the next step — sim stops at simTime on its own
    Simulator::Schedule(Seconds(STEP_TIME_S), &StepCallback);
    return true;
}

// StepCallback: collect per-step stats from DL flows only, send to agent
void StepCallback()
{
    g_monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = g_monitor->GetFlowStats();

    // Reset step stats
    for (auto& u : g_ueStats) {
        u.throughputMbps = 0.0;
        u.queueBytes     = 0.0;
        u.delayMs        = 0.0;
    }

    uint32_t ueIdx = 0;
    for (auto& kv : stats)
    {
        if (ueIdx >= NUM_UE) break;
        auto t   = g_classifier->FindFlow(kv.first);
        // Only DL flows (destination in 7.x.x.x UE subnet)
        if ((t.destinationAddress.Get() >> 24) != 7) continue;

        auto& s  = kv.second;
        auto& us = g_ueStats[ueIdx];

        double   drxBytes = s.rxBytes              - us.rxBytesPrev;
        double   dDelay   = s.delaySum.GetSeconds() - us.delaySumPrev;
        uint64_t dtx      = s.txPackets             - us.txPktsPrev;
        uint64_t drx      = s.rxPackets             - us.rxPktsPrev;

        us.throughputMbps = (drxBytes * 8.0) / STEP_TIME_S / 1e6;
        uint64_t backlog  = (dtx > drx) ? (dtx - drx) : 0;
        us.queueBytes     = (double)backlog * 1252.0;
        us.delayMs        = (drx > 0) ? (dDelay / (double)drx) * 1000.0 : 0.0;

        // Accumulate SINR and Queue for episode average
        us.epAvgSinr  += us.sinrDb;
        us.epAvgQueue += us.queueBytes;
        us.epStepCount++;

        us.rxBytesPrev  = s.rxBytes;
        us.delaySumPrev = s.delaySum.GetSeconds();
        us.txPktsPrev   = s.txPackets;
        us.rxPktsPrev   = s.rxPackets;
        ++ueIdx;
    }

    // Update running max queue
    double sysTh = 0.0;
    for (auto& u : g_ueStats) {
        sysTh += u.throughputMbps;
        // Cap per-UE step throughput to max offered load
        u.throughputMbps = std::min(u.throughputMbps, 10.24);
        if (u.queueBytes > g_maxQueueBytes) g_maxQueueBytes = u.queueBytes;
    }

    // g_stepCounter and g_totalSteps are incremented here but g_totalSteps is
    // moved to AFTER NotifyCurrentState so GetGameOver() reads the post-increment
    // value only on the *next* step query — guaranteeing step 10 of every episode
    // (including the last) fully executes before the agent sees gameOver=true.
    g_stepCounter++;

    // ── Episode boundary — print results and randomize positions ─────────────
    if (g_stepCounter >= STEPS_PER_EP)
    {
        g_episodeCount++;
        FlowMonitor::FlowStatsContainer epStats = g_monitor->GetFlowStats();
        uint32_t ei = 0;
        for (auto& kv : epStats)
        {
            if (ei >= NUM_UE) break;
            auto t = g_classifier->FindFlow(kv.first);
            if ((t.destinationAddress.Get() >> 24) != 7) continue;
            auto& u = g_ueStats[ei];
            double   rxB  = kv.second.rxBytes             - u.epRxBytesStart;
            double   dD   = kv.second.delaySum.GetSeconds()- u.epDelaySumStart;
            int64_t  dtx  = (int64_t)kv.second.txPackets  - (int64_t)u.epTxPktsStart;
            int64_t  drxp = (int64_t)kv.second.rxPackets  - (int64_t)u.epRxPktsStart;
            dtx  = std::max((int64_t)0, dtx);
            drxp = std::max((int64_t)0, std::min(drxp, dtx));
            u.epThroughput = (rxB * 8.0) / EP_TIME_S / 1e6;
            u.epThroughput = std::min(u.epThroughput, 10.24); // cap at max offered load
            u.epDelay      = (drxp > 0) ? (dD / (double)drxp) * 1000.0 : 0.0;
            u.epTxPkts     = (uint64_t)dtx;
            // Compute episode avg SINR and Queue from accumulated step values
            u.epAvgSinr  = (u.epStepCount > 0) ? u.epAvgSinr  / u.epStepCount : u.sinrDb;
            u.epAvgQueue = (u.epStepCount > 0) ? u.epAvgQueue / u.epStepCount : 0.0;
            // Reset accumulators for next episode
            u.epStepCount = 0;
            u.epAvgSinr   = 0.0; 
            u.epAvgQueue  = 0.0; 
            u.epRxPkts     = (uint64_t)drxp;
            // Reset snapshot
            u.epRxBytesStart  = kv.second.rxBytes;
            u.epDelaySumStart  = kv.second.delaySum.GetSeconds();
            u.epTxPktsStart    = kv.second.txPackets;
            u.epRxPktsStart    = kv.second.rxPackets;
            ++ei;
        }
        WriteEpisodeStats(g_episodeCount);

        // Randomize UE positions for next episode
        if (g_randAngle && g_randRadius)
            for (uint32_t i = 0; i < g_ueNodes.GetN(); ++i)
            {
                double r     = g_randRadius->GetValue();
                double theta = g_randAngle->GetValue();
                g_ueNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
                    Vector(g_gnbPos.x + r*std::cos(theta),
                           g_gnbPos.y + r*std::sin(theta), 1.5));
            }

        g_stepCounter = 0;
    }

    // NotifyCurrentState fires BEFORE g_totalSteps is incremented.
    // GetGameOver() is called inside NotifyCurrentState — it will see the
    // OLD (pre-increment) value.  After the ZMQ round-trip returns, we
    // increment so the NEXT call to GetGameOver sees the updated count.
    // Result: step 10 of the final episode is fully processed and episode
    // stats are written before the agent ever receives gameOver=true.
    g_openGym->NotifyCurrentState();
    g_totalSteps++; 
}

// ─────────────────────────────────────────────────────────────────────────────
//  main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    uint16_t gNbNum       = 1;
    uint16_t ueNumPergNb  = 6;
    bool     logging      = false;

    Time simTime         = Seconds(SIM_TIME_S);
    Time udpAppStartTime = MilliSeconds(400);

    uint16_t numerology       = 1;
    double   centralFrequency = 3.5e9;
    double   bandwidth        = 100e6;
    double   totalTxPower     = 43;
    bool     enableOfdma      = true;
    uint16_t mcsTable         = 2;

    uint32_t udpPacketSizeBe = 1252;
    uint32_t lambdaBe        = 1000;

    uint32_t openGymPort = 5555;
    uint32_t simSeed     = 42;

    CommandLine cmd;
    cmd.AddValue("gNbNum",           "gNBs",              gNbNum);
    cmd.AddValue("ueNumPergNb",      "UEs/gNB",           ueNumPergNb);
    cmd.AddValue("logging",          "Enable logging",    logging);
    cmd.AddValue("numerology",       "NR numerology",     numerology);
    cmd.AddValue("centralFrequency", "Frequency",         centralFrequency);
    cmd.AddValue("bandwidth",        "Bandwidth",         bandwidth);
    cmd.AddValue("totalTxPower",     "TX power (dBm)",    totalTxPower);
    cmd.AddValue("simTag",           "Output tag",        g_simTag);
    cmd.AddValue("outputDir",        "Output dir",        g_outputDir);
    cmd.AddValue("enableOfdma",      "OFDMA scheduler",   enableOfdma);
    cmd.AddValue("openGymPort",      "ns3gym port",       openGymPort);
    cmd.AddValue("simSeed",          "Random seed",       simSeed);
    cmd.Parse(argc, argv);

    if (logging)
    {
        LogLevel ll = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME |
                                 LOG_PREFIX_NODE | LOG_LEVEL_INFO);
        LogComponentEnable("EmbbDqnEnv",          ll);
        LogComponentEnable("NrMacSchedulerOfdma", ll);
    }

    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(999999999)); 
    RngSeedManager::SetSeed(12345 + time(NULL) % 100000 + simSeed);
    RngSeedManager::SetRun(simSeed);

    // ── ns3gym ────────────────────────────────────────────────────────────────
    g_openGym = CreateObject<OpenGymInterface>(openGymPort);
    g_openGym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    g_openGym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    g_openGym->SetGetObservationCb(MakeCallback(&GetObservation));
    g_openGym->SetGetRewardCb(MakeCallback(&GetReward));
    g_openGym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    g_openGym->SetGetExtraInfoCb(MakeCallback(&GetExtraInfo));
    g_openGym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));

    // ── Scenario ──────────────────────────────────────────────────────────────
    int64_t randomStream = 1;
    GridScenarioHelper gridScenario;
    gridScenario.SetRows(1);
    gridScenario.SetColumns(gNbNum);
    gridScenario.SetHorizontalBsDistance(500);
    gridScenario.SetVerticalBsDistance(500);
    gridScenario.SetBsHeight(25);
    gridScenario.SetUtHeight(1.5);
    gridScenario.SetSectorization(GridScenarioHelper::SINGLE);
    gridScenario.SetBsNumber(gNbNum);
    gridScenario.SetUtNumber(ueNumPergNb * gNbNum);
    gridScenario.SetScenarioHeight(500);
    gridScenario.SetScenarioLength(500);
    randomStream += gridScenario.AssignStreams(randomStream);
    gridScenario.CreateScenario();

    Ptr<Node> gnb = gridScenario.GetBaseStations().Get(0);
    MobilityHelper mobilityGnb;
    mobilityGnb.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityGnb.Install(gnb);
    gnb->GetObject<MobilityModel>()->SetPosition(Vector(250.0, 250.0, 25.0));

    NodeContainer ueVoiceContainer = gridScenario.GetUserTerminals();
    MobilityHelper mobilityUe;
    mobilityUe.SetMobilityModel("ns3::RandomWalk2dMobilityModel");
    mobilityUe.Install(ueVoiceContainer);

    // Store for re-randomization
    g_ueNodes    = ueVoiceContainer;
    g_gnbPos     = gnb->GetObject<MobilityModel>()->GetPosition();
    g_randAngle  = CreateObject<UniformRandomVariable>();
    g_randAngle->SetAttribute("Min", DoubleValue(0.0));
    g_randAngle->SetAttribute("Max", DoubleValue(2.0 * M_PI));
    g_randRadius = CreateObject<UniformRandomVariable>();
    g_randRadius->SetAttribute("Min", DoubleValue(50.0));
    g_randRadius->SetAttribute("Max", DoubleValue(250.0)); 

    // Initial UE positions
    std::cout << "Initial UE positions:\n";
    for (uint32_t i = 0; i < ueVoiceContainer.GetN(); ++i)
    {
        double r     = g_randRadius->GetValue();
        double theta = g_randAngle->GetValue();
        double x     = g_gnbPos.x + r * std::cos(theta);
        double y     = g_gnbPos.y + r * std::sin(theta);
        ueVoiceContainer.Get(i)->GetObject<MobilityModel>()
                                ->SetPosition(Vector(x, y, 1.5));
        double dist = std::sqrt((x - g_gnbPos.x) * (x - g_gnbPos.x) +
                                (y - g_gnbPos.y) * (y - g_gnbPos.y));
        std::cout << "  UE" << (i+1) << ": " << std::fixed
                  << std::setprecision(1) << dist << " m from gNB\n";
    }

    // ── NR helper ─────────────────────────────────────────────────────────────
    Ptr<NrPointToPointEpcHelper>  epcHelper =
        CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper>   idealBeamformingHelper =
        CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(epcHelper);

    nrHelper->SetGnbPhyAttribute("NoiseFigure", DoubleValue(5.0));
    nrHelper->SetUePhyAttribute("NoiseFigure",  DoubleValue(9.0));
    nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod",
                       TimeValue(MilliSeconds(0)));
    nrHelper->SetChannelConditionModelAttribute("UpdatePeriod",
                                                TimeValue(MilliSeconds(0)));

    // PF scheduler as base — DQN controls behavior via µ/φ in RB metric Eq.(17)
    std::string schedulerStr = "ns3::NrMacScheduler" +
        std::string(enableOfdma ? "Ofdma" : "Tdma") + "PF";
    std::cout << "Base Scheduler: " << schedulerStr << " (µ/φ controlled by DQN)\n";
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName(schedulerStr));

    std::string errorModel = "ns3::NrEesmIrT" + std::to_string(mcsTable);
    nrHelper->SetDlErrorModel(errorModel);
    nrHelper->SetUlErrorModel(errorModel);
    nrHelper->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nrHelper->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
        TypeIdValue(CellScanBeamforming::GetTypeId()));

    nrHelper->SetUeAntennaAttribute("NumRows",    UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
        PointerValue(CreateObject<ThreeGppAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows",    UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
        PointerValue(CreateObject<ThreeGppAntennaModel>()));

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    OperationBandInfo band;
    CcBwpCreator::SimpleOperationBandConf bandConf(
        centralFrequency, bandwidth, 1, BandwidthPartInfo::UMa);//for NLos,LoS
    bandConf.m_numBwp = 1;
    band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    nrHelper->InitializeOperationBand(&band);
    allBwps = CcBwpCreator::GetAllBwps({band});

    double txPow = std::pow(10.0, totalTxPower / 10.0);
    Packet::EnableChecking();
    Packet::EnablePrinting();
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(0));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE",  UintegerValue(0));

    NetDeviceContainer enbNetDev =
        nrHelper->InstallGnbDevice(gridScenario.GetBaseStations(), allBwps);
    NetDeviceContainer ueVoiceNetDev =
        nrHelper->InstallUeDevice(ueVoiceContainer, allBwps);
    g_ueNetDev = ueVoiceNetDev;

    randomStream += nrHelper->AssignStreams(enbNetDev,      randomStream);
    randomStream += nrHelper->AssignStreams(ueVoiceNetDev,  randomStream);

    for (uint32_t i = 0; i < enbNetDev.GetN(); ++i)
    {
        nrHelper->GetGnbPhy(enbNetDev.Get(i), 0)->SetAttribute(
            "Numerology", UintegerValue(numerology));
        nrHelper->GetGnbPhy(enbNetDev.Get(i), 0)->SetAttribute(
            "TxPower", DoubleValue(10.0 * std::log10(txPow)));
        DynamicCast<NrGnbNetDevice>(enbNetDev.Get(i))->UpdateConfig();
    }
    for (auto it = ueVoiceNetDev.Begin(); it != ueVoiceNetDev.End(); ++it)
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();

    // ── SINR trace ────────────────────────────────────────────────────────────
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/NrUePhy/DlDataSinr",
        MakeCallback(&SinrTrace));

    // ── Internet / IP ─────────────────────────────────────────────────────────
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("20Gb/s")));
    p2ph.SetDeviceAttribute("Mtu",      UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay",   TimeValue(Seconds(0.001)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper       ipv4h;
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    ipv4h.Assign(internetDevices);

    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    internet.Install(ueVoiceContainer);
    Ipv4InterfaceContainer ueVoiceIpIface =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueVoiceNetDev));

    for (uint32_t j = 0; j < ueVoiceContainer.GetN(); ++j)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(
                ueVoiceContainer.Get(j)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(
            epcHelper->GetUeDefaultGatewayAddress(), 1);
    }
    nrHelper->AttachToClosestEnb(ueVoiceNetDev, enbNetDev);

    // ── Traffic ───────────────────────────────────────────────────────────────
    uint16_t dlPortVoice = 1235;
    ApplicationContainer serverApps;
    UdpServerHelper dlPacketSinkVoice(dlPortVoice);
    serverApps.Add(dlPacketSinkVoice.Install(ueVoiceContainer));

    UdpClientHelper dlClientVoice;
    dlClientVoice.SetAttribute("RemotePort",  UintegerValue(dlPortVoice));
    dlClientVoice.SetAttribute("MaxPackets",  UintegerValue(0xFFFFFFFF));
    dlClientVoice.SetAttribute("PacketSize",  UintegerValue(udpPacketSizeBe));
    dlClientVoice.SetAttribute("Interval",    TimeValue(Seconds(1.0 / lambdaBe)));

    EpsBearer   voiceBearer(EpsBearer::GBR_CONV_VOICE);
    Ptr<EpcTft> voiceTft = Create<EpcTft>();
    EpcTft::PacketFilter dlpfVoice;
    dlpfVoice.localPortStart = dlPortVoice;
    dlpfVoice.localPortEnd   = dlPortVoice;
    voiceTft->Add(dlpfVoice);

    ApplicationContainer clientApps;
    for (uint32_t i = 0; i < ueVoiceContainer.GetN(); ++i)
    {
        dlClientVoice.SetAttribute("RemoteAddress",
            AddressValue(ueVoiceIpIface.GetAddress(i)));
        clientApps.Add(dlClientVoice.Install(remoteHost));
        nrHelper->ActivateDedicatedEpsBearer(
            ueVoiceNetDev.Get(i), voiceBearer, voiceTft);
    }

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    // ── FlowMonitor ───────────────────────────────────────────────────────────
    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueVoiceContainer);
    g_monitor    = flowmonHelper.Install(endpointNodes);
    g_classifier = DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    g_monitor->SetAttribute("DelayBinWidth",      DoubleValue(0.001));
    g_monitor->SetAttribute("JitterBinWidth",     DoubleValue(0.001));
    g_monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    // ── Progress ──────────────────────────────────────────────────────────────
    Time effectiveSimTime = simTime - udpAppStartTime;
    std::cout << "\nSimulation: " << MAX_EPISODES << " episodes × "
              << EP_TIME_S << "s = " << (MAX_EPISODES * EP_TIME_S)
              << "s active traffic\n";
    Simulator::Schedule(udpAppStartTime + effectiveSimTime * 0.25,
        []() { std::cout << "Progress: 25%\n"; });
    Simulator::Schedule(udpAppStartTime + effectiveSimTime * 0.50,
        []() { std::cout << "Progress: 50%\n"; });
    Simulator::Schedule(udpAppStartTime + effectiveSimTime * 0.75,
        []() { std::cout << "Progress: 75%\n"; });

    // ── Start DQN step loop ───────────────────────────────────────────────────
    Simulator::Schedule(udpAppStartTime + Seconds(STEP_TIME_S), &StepCallback);

    // ── Final stats callback — scheduled at simTime - 0.001s ─────────────────
    // Runs INSIDE Simulator::Run() while FlowMonitor is still valid and BEFORE
    // ns3gym ZMQ teardown can kill the process.  This guarantees the final
    // flow results and CSV are always printed regardless of how ns3gym exits.
    double finalPrintTime = SIM_TIME_S - 0.001;
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    double flowDuration = effectiveSimTime.GetSeconds();

    Simulator::Schedule(Seconds(finalPrintTime), [classifier, flowDuration]()
    {
        g_monitor->CheckForLostPackets();
        FlowMonitor::FlowStatsContainer finalStats = g_monitor->GetFlowStats();

        double sumTput = 0, sumDelay = 0, sumPLR = 0;
        uint32_t flowCount = 0;
        double fTput[NUM_UE]={0}, fDelay[NUM_UE]={0}, fPLR[NUM_UE]={0}, fJitter[NUM_UE]={0};
        uint32_t fi = 0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n====== Final Simulation Results ======\n";

        for (auto& kv : finalStats)
        {
            auto t = classifier->FindFlow(kv.first);
            if ((t.destinationAddress.Get() >> 24) != 7) continue;

            auto& s = kv.second;
            uint32_t tx = s.txPackets, rx = s.rxPackets;
            double plr  = (tx > 0 && tx >= rx) ? (tx - rx) * 100.0 / tx : 0.0;
            double tput = (rx > 0) ? s.rxBytes * 8.0 / flowDuration / 1e6 : 0.0;
            double dms  = (rx > 0) ? 1000.0 * s.delaySum.GetSeconds()  / rx : 0.0;
            double jms  = (rx > 0) ? 1000.0 * s.jitterSum.GetSeconds() / rx : 0.0;
            double txOff = s.txBytes * 8.0 / flowDuration / 1e6;

            std::cout << "UE (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
            std::cout << " Tx Packets:  " << tx           << "\n";
            std::cout << " Tx Bytes:    " << s.txBytes    << "\n";
            std::cout << " TxOffered:   " << txOff        << " Mbps\n";
            std::cout << " Rx Bytes:    " << s.rxBytes    << "\n";
            std::cout << " Throughput:  " << tput         << " Mbps\n";
            std::cout << " Delay:  " << dms          << " ms\n";
            std::cout << " Jitter: " << jms          << " ms\n";
            std::cout << " Packet Loss: " << plr          << " %\n\n";

            sumTput += tput; sumDelay += dms; sumPLR += plr;
            if (fi < NUM_UE) {
                fTput[fi]=tput; fDelay[fi]=dms; fJitter[fi]=jms; fPLR[fi]=plr;
                fi++;
            }
            flowCount++;
        }

        double avgPlr = (flowCount > 0) ? sumPLR / flowCount : 0.0;
        std::cout << "Mean flow throughput: " << (flowCount > 0 ? sumTput  / flowCount : 0.0) << " Mbps\n";
        std::cout << "Mean flow delay:      " << (flowCount > 0 ? sumDelay / flowCount : 0.0) << " ms\n";
        std::cout << "Average Packet Loss Rate: " << avgPlr << " %\n";
        std::cout << "======================================\n";
        std::cout.flush();

        // ── Final summary CSV ─────────────────────────────────────────────────
        std::string csvFinal = g_outputDir + "/results_final_02" + g_simTag + ".csv";
        std::ofstream cf(csvFinal, std::ofstream::out | std::ofstream::trunc);
        if (cf.is_open())
        {
            cf << std::fixed << std::setprecision(6);
            cf << "Metric,UE1,UE2,UE3,UE4,UE5,UE6,Mean\n";
            auto row = [&](const std::string& name, double v[]) {
                double s = 0; cf << name;
                for (uint32_t i = 0; i < NUM_UE; i++) { cf << "," << v[i]; s += v[i]; }
                cf << "," << s / NUM_UE << "\n";
            };
            row("Throughput_Mbps", fTput);
            row("Delay_ms",    fDelay);
            row("Jitter_ms",   fJitter);
            row("PLR_pct",         fPLR);
            cf << "\nStat,Value\n";
            cf << "TotalEpisodes,"           << g_episodeCount << "\n";
            cf << "MeanFlowThroughput_Mbps," << (flowCount > 0 ? sumTput  / flowCount : 0.0) << "\n";
            cf << "MeanFlowDelay_ms,"        << (flowCount > 0 ? sumDelay / flowCount : 0.0) << "\n";
            cf << "MeanPLR_pct,"             << avgPlr << "\n";
            cf.close();
            std::cout << "Final summary CSV saved → " << csvFinal << "\n";
            std::cout.flush();
        }
    });

    Simulator::Stop(simTime);
    Simulator::Run();

    g_openGym->NotifySimulationEnd();
    Simulator::Destroy();
    return 0;
}
