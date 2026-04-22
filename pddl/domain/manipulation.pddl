; PDDL domain for tabletop manipulation (Phase 1 — arm only, no navigation)

(define (domain manipulation)
  (:requirements :strips :typing)

  (:types
    object location - entity
  )

  (:predicates
    (on ?obj - object ?loc - location)     ; object is at location
    (holding ?obj - object)               ; gripper holds object
    (gripper-empty)                       ; gripper is free
    (reachable ?loc - location)           ; location is within arm reach
  )

  (:action pick
    :parameters (?obj - object ?loc - location)
    :precondition (and (on ?obj ?loc) (gripper-empty) (reachable ?loc))
    :effect (and (holding ?obj) (not (gripper-empty)) (not (on ?obj ?loc)))
  )

  (:action place
    :parameters (?obj - object ?loc - location)
    :precondition (and (holding ?obj) (reachable ?loc))
    :effect (and (on ?obj ?loc) (gripper-empty) (not (holding ?obj)))
  )
)
