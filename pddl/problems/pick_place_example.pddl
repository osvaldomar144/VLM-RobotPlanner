; Example problem: pick red_cup from table_a and place it on table_b

(define (problem pick-place-example)
  (:domain manipulation)

  (:objects
    red_cup - object
    table_a table_b - location
  )

  (:init
    (on red_cup table_a)
    (gripper-empty)
    (reachable table_a)
    (reachable table_b)
  )

  (:goal
    (on red_cup table_b)
  )
)
