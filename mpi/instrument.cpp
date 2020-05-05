/*
 * instrument.cpp
 *
 * Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)
 *
 * Instrumenation Implementation
 *
 * Credit: 15418-s20 Assignment source code
 */
#include "instrument.h"

#include <iostream>
#include <ctime>
#include <ratio>
#include <string>
#include <chrono>

using namespace std::chrono;

using Clock=system_clock;
//using Time=system_clock;

static char *activity_name[ACTIVITY_COUNT] = {"unknown", "setup", "propogate", "predict", "comm"};

static bool initialized = false;

static bool tracking = false;
static std::time_t global_start_time;

#define MAXDEPTH 20

static activity_t activity_stack[MAXDEPTH];
static int stack_level = 0;

static std::time_t current_start_time;

static double accum[ACTIVITY_COUNT];

void track_activity(bool enable) {
    tracking = enable;
}

static void init_instrument() {
    if (!tracking)
    return;
    if (initialized)
    return;
    initialized = true;
    Clock::time_point now = Clock::now();
    global_start_time = Clock::to_time_t(now);
    current_start_time = Clock::to_time_t(now);
    int a;
    for (a = 0; a < ACTIVITY_COUNT; a++) {
    accum[a] = 0.0;
    }
    stack_level = 0;
    activity_stack[stack_level] = ACTIVITY_NONE;
}

void start_activity(activity_t a) {
    if (!tracking)
    return;
    init_instrument();
    int olda = activity_stack[stack_level];
    Clock::time_point new_time = Clock::now();
    accum[olda] += difftime(Clock::to_time_t(new_time), current_start_time);
    current_start_time = Clock::to_time_t(new_time);
    activity_stack[++stack_level] = a;
    if (stack_level >= MAXDEPTH) {
      fprintf(stderr, "Runaway instrumentation activity stack.  Disabling\n");
      tracking = false;
      return;
    }
}

void finish_activity(activity_t a) {
    if (!tracking)
    return;
    init_instrument();
    int olda = activity_stack[stack_level];
    if (a != olda) {
    fprintf(stderr, "Warning.  Started activity %s, but now finishing activity %s.  Disabling\n",
        activity_name[olda], activity_name[a]);
      tracking = false;
      return;
    }
    auto new_time = Clock::now();
    auto diff =  difftime(Clock::to_time_t(new_time), current_start_time);
    accum[olda] += diff;
    current_start_time = Clock::to_time_t(new_time);
    stack_level--;
    if (stack_level < 0) {
      fprintf(stderr, "Warning, popped off bottom of instrumentation activity stack.  Disabling\n");
      tracking = false;
      return;
    }

}


void show_activity(FILE *f) {
  if (!tracking)
    return;
  init_instrument();
  int a;
  auto new_time = Clock::to_time_t(Clock::now());
  double elapsed = difftime(new_time, global_start_time);
  double unknown = elapsed;
  for (a = 1; a < ACTIVITY_COUNT; a++)
    unknown -= accum[a];
  accum[0] = unknown;


  for (a = 0; a < ACTIVITY_COUNT; a++) {
    if (accum[a] == 0.0)
        continue;
    double ms = accum[a] * 1000.0;
    double pct = accum[a] / elapsed * 100.0;
    fprintf(f, "    %8d ms    %5.1f %%    %s\n", (int) ms, pct, activity_name[a]);
  }
  fprintf(f, "    %8d ms    %5.1f %%    elapsed\n", (int) (elapsed * 1000.0), 100.0);
}


