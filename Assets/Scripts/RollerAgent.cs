using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


public class RollerAgent : Agent
{
    Rigidbody rBody;
    public float forceMultiplier = 10;
    public Transform Target;

    public bool trainingMode;

    void Start () {
        rBody = GetComponent<Rigidbody>();
    }

    
    public override void OnEpisodeBegin()
    {
       // If the Agent fell, zero its momentum
        // if (this.transform.localPosition.y < 0)
        // {
        //     this.rBody.angularVelocity = Vector3.zero;
        //     this.rBody.velocity = Vector3.zero;
        //     this.transform.localPosition = new Vector3( 0, 0.5f, 0);
        // }

        // Zero out velocities so that movement stops before a new episode begins
        // this.rBody.angularVelocity = Vector3.zero;
        // this.rBody.velocity = Vector3.zero;

        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 8 - 4,
                                           Random.value * 8 + 1,
                                           Random.value * 8 - 4);
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Target and Agent positions
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);

        // Agent velocity
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
        sensor.AddObservation(rBody.velocity.y);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        controlSignal.y = actionBuffers.ContinuousActions[2];
        rBody.AddForce(controlSignal * forceMultiplier);

        // // Rewards
        // float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // // Reached target
        // if (trainingMode && distanceToTarget < 1f)
        // {
        //     SetReward(1.0f);
        //     EndEpisode();
        // }

        // Fell off platform
        // else if (this.transform.localPosition.y < 0)
        // {
        //     EndEpisode();
        // }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (trainingMode && collision.collider.CompareTag("floor"))
        {
            // Collided with the area boundary, give a negative reward
            AddReward(-.5f);
        }

        // Reached target
        if (trainingMode && collision.collider.CompareTag("target"))
        {
            SetReward(1.0f);
            EndEpisode();
        }
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // continuousActionsOut[0] = Input.GetAxis("Horizontal");
        // continuousActionsOut[1] = Input.GetAxis("Vertical");

        // Create placeholders for all movement/turning
        Vector3 forward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;

        // Convert keyboard inputs to movement and turning
        // All values should be between -1 and +1

        // Forward/backward
        if (Input.GetKey(KeyCode.UpArrow)) forward = transform.forward;
        else if (Input.GetKey(KeyCode.DownArrow)) forward = -transform.forward;

        // Left/right
        if (Input.GetKey(KeyCode.LeftArrow)) left = -transform.right;
        else if (Input.GetKey(KeyCode.RightArrow)) left = transform.right;

        // Up/down
        if (Input.GetKey(KeyCode.W)) up = transform.up;
        else if (Input.GetKey(KeyCode.S)) up = -transform.up;

        // Combine the movement vectors and normalize
        Vector3 combined = (forward + left + up).normalized;

        // Add the 3 movement values, pitch, and yaw to the actionsOut array
        continuousActionsOut[0] = combined.x;
        continuousActionsOut[1] = combined.y;
        continuousActionsOut[2] = combined.z;
    }
}

