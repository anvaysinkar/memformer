#include "pin.H"
#include <fstream>
#include <iostream>

KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "trace.out", "output file");

static std::ofstream TraceFile;
static UINT64 count = 0;
static const UINT64 MAX_RECORDS = 2000000;

VOID RecordMemAccess(ADDRINT pc, ADDRINT addr) {
    if (count++ < MAX_RECORDS)
        TraceFile << std::hex << pc << " " << addr << "\n";
}

VOID Instruction(INS ins, VOID *v) {
    UINT32 memOps = INS_MemoryOperandCount(ins);
    for (UINT32 memOp = 0; memOp < memOps; memOp++) {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemAccess,
            IARG_INST_PTR, IARG_MEMORYOP_EA, memOp, IARG_END);
    }
}

VOID Fini(INT32 code, VOID *v) { TraceFile.close(); }

INT32 Usage() {
    std::cerr << "Memory trace collector\n";
    return -1;
}

int main(int argc, char *argv[]) {
    if (PIN_Init(argc, argv)) return Usage();
    TraceFile.open(KnobOutputFile.Value().c_str());
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;
}
