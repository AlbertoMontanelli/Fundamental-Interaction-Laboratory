#pragma once
#include <vector>
#include "Framework.hh"
#include <iostream>
#include <algorithm>

class Trigger{
public:
  enum triggerType { cosmic, accidental };
  double MinSuperposition;
  double CosmicRate;
  double NextAccidentalTriggerTime;
  double NextCosmicTriggerTime;
  double RunLength;
  
  vector<Detector *> Detectors;

  Trigger(double minSuperposition = 2*ns_u, double cosmicRate = 10 ):
    MinSuperposition(minSuperposition),
    CosmicRate( cosmicRate ),
    NextAccidentalTriggerTime(0.),
    NextCosmicTriggerTime(0.),
    RunLength(1000.)
  {
    //    UpdateNextCosmicTrigger();
    //    UpdateNextAccidentalTrigger();
  };

  void AddDetector ( Detector * det){
    Detectors.push_back( det );
    sort( Detectors.begin(), Detectors.end(), [] (const Detector *d1, const Detector *d2){ return d1->GetNextFakeHitTime() > d2->GetNextFakeHitTime();});
        
  }
  
  tuple<double,triggerType> NextTrigger(void);

  double UpdateNextCosmicTrigger(void); // returns the time of the next cosmic trigger
  double UpdateNextAccidentalTrigger(void); // returns the time of the next accidental trigger
  
};
