/*
 * instrument.h
 *
 * Authors: Joel Anyanti, Edward Lucero (Carnegie Mellon University)
 *
 * Instrumenation Code to keep track of performance
 * TODO: Description
 *
 * Credit: 15418-s20 Assignment source code
 */
#ifndef INSTRUMENT_H

// #ifndef TRACK
#define TRACK 1
// #endif

#include <stdio.h>
#include <stdbool.h>


typedef enum { ACTIVITY_NONE, ACTIVITY_SETUP, ACTIVITY_TRAIN, ACTIVITY_COMM, ACTIVITY_PREDICT, ACTIVITY_COUNT } activity_t;

void track_activity(bool enable);

void start_activity(activity_t a);
void finish_activity(activity_t a);
void show_activity(FILE *f);

#if TRACK
#define TRACK_ACTIVITY(e) track_activity(e);
#define START_ACTIVITY(a) start_activity(a);
#define FINISH_ACTIVITY(a) finish_activity(a);
#define SHOW_ACTIVITY(f) show_activity(f);
#else
#define TRACK_ACTIVITY(e)  /* Optimized out */
#define START_ACTIVITY(a)   /* Optimized out */
#define FINISH_ACTIVITY(a)  /* Optimized out */
#define SHOW_ACTIVITY(f)  /* Optimized out */
#endif

#define INSTRUMENT_H
#endif
